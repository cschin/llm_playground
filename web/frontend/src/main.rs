#![allow(non_snake_case)]
// import the prelude to get access to the `rsx!` macro and the `Scope` and `Element` types
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct QueryText {
    topn: u64,
    text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentRecord {
    score: f32,
    file_name: Option<String>,
    url: Option<String>,
    document_id: Option<usize>,
    section_id: Option<usize>,
    chunk_id: Option<usize>,
    keywords: Option<String>,
    text: Option<String>,
}

pub fn base_url() -> String {
    web_sys::window().unwrap().location().origin().unwrap()
}

fn main() {
    // launch the web app
    dioxus_web::launch(App);
    wasm_logger::init(wasm_logger::Config::default());
}

// create a component that renders a div with the text "Hello, world!"
fn App(cx: Scope) -> Element {
    let diags = use_ref(cx, || Vec::<(String, Vec<DocumentRecord>)>::new());
    let current_input = use_state(cx, || "Tell me about Floating-Harbor Syndrome".to_string());

    cx.render(rsx! {
        div { class: "flex flex-col p-12 justify-center h-full",
            div { class: "flex justify-center border-solid border-2 rounded-lg border-indigo-600",
                p { class: "mb-2 mt-0 text-3xl font-medium leading-tight text-primary",
                    "Ask Anything about GeneReviews"
                }
            }

            div { class: "flex flex-col flex-col-reverse h-full",
                div { class: "flex flex-row p-2 justify-center border-solid border-2 rounded-lg border-indigo-600 justify-end",
                    textarea {
                        class: "px-3 py-1",
                        cols: 120,
                        rows: 3,
                        // we tell the component what to render
                        value: "{current_input}",
                        // and what to do when the value changes
                        oninput: move |evt| current_input.set(evt.value.clone())
                    }
                    div { class: "px-3 py-1",
                        div {
                            class: "px-2 py-1 h-full middle none rounded-lg bg-blue-600 text-white",
                            onclick: move |_evt| {
                                let input = current_input.get().clone();
                                post_query(cx, "post_query_for_similarity_search".to_string(), &input, diags);
                            },
                            "Find Similar Text"
                        }
                    }
                    div { class: "px-3 py-1",
                        div {
                            class: "px-2 py-1 h-full middle none rounded-lg bg-blue-600 text-white",
                            onclick: move |_evt| {
                                let input = current_input.get().clone();
                                post_query(cx, "post_query_for_summary_of_a_topic".to_string(), &input, diags);

                            },
                            "Write a Summary "
                        }
                    }
                    div { class: "px-3 py-1",
                        div {
                            class: "px-2 py-1 h-full middle none rounded-lg bg-blue-600 text-white",
                            onclick: move |_evt| {
                                let input = current_input.get().clone();
                                post_query(cx, "post_query_for_answer_of_a_question".to_string(), &input, diags);
                            },
                            "Get an Answer"
                        }
                    }
                }
                div { class: "container h-full py-8 px-2 border-solid border-2 rounded-lg border-indigo-600 overscroll-contain overflow-auto",
                    Dialogs(cx, diags)
                }
            }
        }
    })
}

fn Dialogs<'a>(cx: Scope<'a>, diags: &'a UseRef<Vec<(String, Vec<DocumentRecord>)>>) -> Element<'a> {
    cx.render(rsx! {
        div { class: "flex flex-col",
            diags.read().iter().map( move |(input, output)| {
                let input = input.clone();
                let output = output.clone();
                rsx! {
                    div {
                        div {
                            p {"## INPUT" } 
                            br {}
                            p {"{input}"}
                            br {}
                        }
                        div { p {"## OUTPUT START -------------------"}}
                        br {}
                        output.iter().map( move |output|
                        rsx! {
                            
                            div {
                            if output.url.is_some() {
                                let url = output.url.clone().unwrap(); 
                                rsx! {pre {"URL:" a {href:"{url}", "{url}"}}}
                            }
                            br {}
                            if output.keywords.is_some() {
                                let keywords = output.keywords.clone().unwrap(); 
                                rsx! {pre {"KEYWORD: {keywords}"}}
                            }
                            br {}
                            if output.text.is_some() {
                                let text = output.text.clone().unwrap(); 

                                rsx!{ p{ "TEXT:\n" }
                                text.split("\n").into_iter().map(|text|  rsx! {p {text} br {}}) }
                            }
                            br {}

                            }  
                        })
                        div { p {"OUTPUT END ==================="}}
                    }
                }
            })
        }
    })
}

fn post_query<'a, T>(
    cx: Scope<'a, T>,
    entry: String,
    query: &'a String,
    records: &'a UseRef<Vec<(String, Vec<DocumentRecord>)>>,
) -> () {
    let query = QueryText {
        text: query.clone(),
        topn: 3,
    };
    let records = records.to_owned();
    cx.spawn({
        async move {
            let client = reqwest::Client::new();
            let url = base_url() + "/api/" + &entry[..];
            let response = client
                .post(url)
                .json(&query)
                .send()
                .await
                .unwrap()
                .json::<Option<Vec<DocumentRecord>>>()
                .await;
            log::debug!("{:?}", response);
            match response {
                Ok(val) => {
                    if val.is_some() {
                        records.write().push((query.text, val.unwrap()));
                    };
                }
                Err(e) => {
                    log::debug!("{:?}", e);
                }
            };
        }
    })
}
