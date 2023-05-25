use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use glob::glob;
use llm_chain::text_splitter::*;
use llm_chain::traits::Embeddings;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[clap(name = "get_open_ai_embed_vec")]
//#[clap(author, version)]
//#[clap(about, long_about = None)]
struct CmdOptions {
    /// the path to the directory that contains the collection of nxml files
    path_to_nxm_files: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct DocumentRecord {
    file_name: String,
    document_id: usize,
    section_id: usize,
    chunk_id: usize,
    text: String,
    embedding_vec: Vec<f32>,
}

fn get_nxml_text(path: &PathBuf) -> Vec<String> {
    let mut file = BufReader::new(File::open(path).expect("file open error"));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("file reading error");
    let text = String::from_utf8_lossy(&buf[..]).to_owned();
    let mut reader = Reader::from_str(&text);
    reader.trim_text(true);
    let mut buf = Vec::new();
    let mut txt = Vec::new();
    let mut sec_txt = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"title" => {
                    let span = reader.read_text(e.name()).expect("err");
                    let just_string = remove_xml_tags(&span, "");
                    sec_txt.push(
                        [String::from("title: =="), just_string, String::from("==")].join(" "),
                    );
                }
                b"p" => {
                    let span = reader.read_text(e.name()).expect("err");
                    let just_string = remove_xml_tags(&span, "");
                    sec_txt.push(
                        [
                            String::from("content: START"),
                            just_string,
                            String::from("END"),
                        ]
                        .join(" "),
                    );
                }
                b"table" => {
                    let span = reader.read_text(e.name()).expect("err");
                    let just_string = remove_xml_tags(&span, "");
                    sec_txt.push([String::from("table:\n"), just_string].join(""));
                }
                b"list" => {
                    let span = reader.read_text(e.name()).expect("err");
                    let just_string = remove_xml_tags(&span, " ");
                    sec_txt.push([String::from("list:\n"), just_string].join(""));
                }
                b"ref" => {
                    let span = reader.read_text(e.name()).expect("err");
                    let just_string = remove_xml_tags(&span, &" ");
                    sec_txt.push([String::from("reference: "), just_string].join(""));
                }
                _ => (),
            },
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"sec" => {
                    txt.push(sec_txt.join("\n"));
                    //txt.push(String::from("\n\n"));
                    sec_txt.clear();
                }
                _ => (),
            },
            Ok(Event::Eof) => break,
            _ => (),
        }
        buf.clear();
    }
    txt
}

fn remove_xml_tags(xml: &str, sep: &str) -> String {
    let mut reader = Reader::from_str(xml);
    reader.trim_text(true);
    let mut buf = Vec::new();
    let mut txt = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            Ok(Event::Start(_)) => continue,
            Ok(Event::Text(e)) => {
                txt.push(String::from_utf8_lossy(e.as_ref()).into_owned());
            }
            Ok(Event::Eof) => break,
            _ => (),
        }
        buf.clear();
    }

    txt.join(sep)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    //CmdOptions::command().version(VERSION_STRING).get_matches();
    let args = CmdOptions::parse();
    let mut document_id_tuple: Vec<(String, usize)> = Vec::new();
    let mut document_id = 0_usize;
    let path_to_nxm_files = args.path_to_nxm_files + "/*.nxml";
    for e in glob(&path_to_nxm_files).expect("Failed to read glob pattern") {
        let path = e.unwrap();
        let file_name = path
            .as_path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        document_id_tuple.push((file_name.clone(), document_id));

        //println!("Load {path:?}");
        let doc = get_nxml_text(&path);
        let mut all_chunk = Vec::new();
        doc.into_iter()
            .enumerate()
            .filter(|(_section_id, section)| section.len() > 0)
            .for_each(|(section_id, section)| {
                let splitter = NaiveWhitespaceSplitter;
                let max_tokens_per_chunk = 256;
                let chunk_overlap = 32;
                let chunks = splitter
                    .split_text(&section, max_tokens_per_chunk, chunk_overlap)
                    .expect("text split error")
                    .into_iter()
                    .filter_map(|chunk| {
                        let chunk = chunk.trim();
                        if chunk.len() > 5 {
                            Some(chunk.to_string())
                        } else {
                            None
                        }
                    })
                    .enumerate()
                    .map(|(chunk_id, chunk)| {
                        (file_name.clone(), document_id, section_id, chunk_id, chunk)
                    })
                    .collect::<Vec<_>>();
                //println!("{} {:?}", chunks.len(), chunks);
                all_chunk.extend(chunks);
            });

        let all_chunk_text = all_chunk
            .iter()
            .map(|chunk| chunk.4.clone())
            .collect::<Vec<String>>();
        if all_chunk_text.len() == 0 {
            continue
        } 

        let embeddings = llm_chain_openai::embeddings::Embeddings::default();
        let embedded_vecs = embeddings.embed_texts(all_chunk_text).await.unwrap();
        embedded_vecs
            .into_iter()
            .zip(all_chunk)
            .map(|(embedding_vec, chuck_record)| {
                let (file_name, document_id, section_id, chunk_id, text) = chuck_record;
                let r = DocumentRecord {
                    file_name,
                    document_id,
                    section_id,
                    chunk_id,
                    text,
                    embedding_vec,
                };
                let jsonl_record = serde_json::to_string(&r).expect("json conversion fails");
                jsonl_record
            })
            .for_each(|r| println!("{}", r));
        document_id += 1;
    }
}

// fn _main() {
//     for e in glob("./test_doc/gene_NBK1116/*.pdf").expect("Failed to read glob pattern") {
//         let path = e.unwrap();
//         println!("{}", path.clone().display());
//         //let bytes = std::fs::read(path).unwrap();
//         println!("Load {path:?}");
//         let doc = load_pdf(&path).expect("load_pdf fails");

//         let text = get_pdf_text(&doc).expect("get pdf text fails");
//         if !text.errors.is_empty() {
//             eprintln!("{path:?} has {} errors:", text.errors.len());
//             for error in &text.errors[..10] {
//                 eprintln!("{error:?}");
//             }
//         }
//         let json_text = serde_json::to_string_pretty(&text).unwrap();
//         println!("{}", json_text);
//         break;
//     }
// }
