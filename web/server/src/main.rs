mod query_qdrant_db;
use crate::query_qdrant_db::*;
use axum::{
    body::{boxed, Body},
    http::{Response, StatusCode},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use llm_chain::{
    chains::conversation::Chain, executor, output::Output, parameters, prompt, step::Step,
};
use llm_chain_openai::chatgpt::{Model, PerInvocation, PerExecutor};
//use llm_chain_openai::chatgpt::{Executor, Model, PerExecutor, PerInvocation};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::{
    net::{IpAddr, Ipv6Addr},
    path::PathBuf,
    str::FromStr,
};
use tokio::fs;
use tower::{ServiceBuilder, ServiceExt};
use tower_http::cors::Any;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[clap(name = "server", about = "Experimental Server")]
struct Opt {
    /// set the listen addr
    #[clap(short = 'a', long = "addr", default_value = "::1")]
    addr: String,

    /// set the listen port
    #[clap(short = 'p', long = "port", default_value = "3000")]
    port: u16,

    /// set the directory where static files are to be found
    #[clap(long = "static-dir", default_value = "./dist")]
    static_dir: String,
}

#[tokio::main]
async fn main() {
    let opt = Opt::parse();

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "example_tracing_aka_logging=debug,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // build our application with a route
    let app = Router::new()
        .route(
            "/api/post_query_for_similarity_search",
            post(  post_query_for_similarity_search ),
        )
        .route(
            "/api/post_query_for_answer_of_a_question",
            post( post_query_for_answer_of_a_question ),
        )
        .route(
            "/api/post_query_for_summary_of_a_topic",
            post( post_query_for_summary_of_a_topic ),
        )
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                //.allow_origin("http://127.0.0.1:8080".parse::<HeaderValue>().unwrap())
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
        .fallback(get(|req| async move {
            match ServeDir::new(&opt.static_dir).oneshot(req).await {
                Ok(res) => {
                    let status = res.status();
                    match status {
                        StatusCode::NOT_FOUND => {
                            let index_path = PathBuf::from(&opt.static_dir).join("index.html");
                            let index_content = match fs::read_to_string(index_path).await {
                                Err(_) => {
                                    return Response::builder()
                                        .status(StatusCode::NOT_FOUND)
                                        .body(boxed(Body::from("index file not found")))
                                        .unwrap()
                                }
                                Ok(index_content) => index_content,
                            };

                            Response::builder()
                                .status(StatusCode::OK)
                                .body(boxed(Body::from(index_content)))
                                .unwrap()
                        }
                        _ => res.map(boxed),
                    }
                }
                Err(_err) => Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(boxed(Body::from("internal errors")))
                    .expect("error response"),
            }
        }));

    // run it
    let addr = SocketAddr::from((
        IpAddr::from_str(opt.addr.as_str()).unwrap_or(IpAddr::V6(Ipv6Addr::LOCALHOST)),
        opt.port,
    ));
    println!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

#[derive(Serialize, Deserialize, Debug)]
struct QueryText {
    topn: u64,
    text: String,
}

async fn post_query_for_similarity_search(
    Json(query): Json<QueryText>,
) -> Json<Option<Vec<DocumentRecord>>> {
    let return_docs = query_doc_vec_db(&query.text, query.topn);
    match return_docs.await {
        Ok(r) => Json(Some(r)),
        _ => Json(None),
    }
}

async fn post_query_for_answer_of_a_question(
    Json(query): Json<QueryText>,
) -> Json<Option<Vec<DocumentRecord>>> {
    let docs = query_doc_vec_db(&query.text, query.topn).await;
 

    let context = if let Ok(records) = docs {
        records
            .iter()
            .map(|record| {
                let record = record.clone();
                let mut out_strings = Vec::<String>::new();
                if record.keywords.is_some() {
                    out_strings.push(format!("KEYWORDS: {}", record.keywords.unwrap()))
                };
                if record.url.is_some() {
                    out_strings.push(format!("URL: {}", record.url.unwrap()))
                };
                if record.text.is_some() {
                    out_strings.push(format!("CONTENT: {}", record.text.unwrap()))
                };
                out_strings.join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        "".to_string()
    };

    if context.is_empty() {
        return Json(None);
    }

    let model = Model::Other("gpt-4".to_string());
    //let model = Model::ChatGPT3_5Turbo;
    let per_invocation = PerInvocation::new().for_model(model);
    let per_executor = PerExecutor { api_key: None };
    let exec = executor!(chatgpt, per_executor, per_invocation).unwrap();



    let mut chain = Chain::new(
                prompt!(system: "You are an assistant for making scientific recommendations.")).unwrap();
    dbg!(&context); 

    let res = chain
        .send_message(
            Step::for_prompt_template(prompt!(user: "given the context below\n: CONTEXT: {{context}} \n\n answer the question: {{question}}")), 
            // Create a Parameters object with key-value pairs for the placeholders
            &parameters!("context" => &context[..],
                         "question" => &query.text[..]),
            &exec,
        )
        .await
        .unwrap();

    let text = res.primary_textual_output().await;
    let r = DocumentRecord {
        text,
        ..Default::default()
    };
    Json(Some(vec![r]))
}


async fn post_query_for_summary_of_a_topic(
    Json(query): Json<QueryText>,
) -> Json<Option<Vec<DocumentRecord>>> {
    let docs = query_doc_vec_db(&query.text, query.topn).await;
 

    let context = if let Ok(records) = docs {
        records
            .iter()
            .map(|record| {
                let record = record.clone();
                let mut out_strings = Vec::<String>::new();
                if record.keywords.is_some() {
                    out_strings.push(format!("KEYWORDS: {}", record.keywords.unwrap()))
                };
                if record.url.is_some() {
                    out_strings.push(format!("URL: {}", record.url.unwrap()))
                };
                if record.text.is_some() {
                    out_strings.push(format!("CONTENT: {}", record.text.unwrap()))
                };
                out_strings.join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        "".to_string()
    };

    if context.is_empty() {
        return Json(None);
    }

    //let model = Model::Other("text-davinci-002".to_string());
    let model = Model::ChatGPT3_5Turbo;
    let per_invocation = PerInvocation::new().for_model(model);
    let per_executor = PerExecutor { api_key: None };
    let exec = executor!(chatgpt, per_executor, per_invocation).unwrap();



    let mut chain = Chain::new(
                prompt!(system: "You are an assistant for making scientific recommendations.")).unwrap();
    dbg!(&context); 

    let res = chain
        .send_message(
            Step::for_prompt_template(prompt!(user: "given the context below\n: CONTEXT: {{context}} \n\n Please write an essay about the topic {{topic}} with evidences and references")), 
            // Create a Parameters object with key-value pairs for the placeholders
            &parameters!("context" => &context[..],
                         "topic" => &query.text[..]),
            &exec,
        )
        .await
        .unwrap();

    dbg!(res.clone());

    let text = res.primary_textual_output().await;
    let r = DocumentRecord {
        score: None,
        file_name: None,
        url: None,
        document_id: None,
        section_id: None,
        chunk_id: None,
        keywords: None,
        text,
        ..Default::default()
    };
    Json(Some(vec![r]))
}