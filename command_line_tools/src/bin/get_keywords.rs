use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[clap(name = "get_keywords")]
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

fn get_keyword_groups(path: &PathBuf) -> Vec<String> {
    let mut file = BufReader::new(File::open(path).expect("file open error"));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("file reading error");
    let text = String::from_utf8_lossy(&buf[..]);
    let mut reader = Reader::from_str(&text);
    reader.trim_text(true);
    let mut buf = Vec::new();
    let mut txt = Vec::new();
    //let mut sec_txt = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            Ok(Event::Start(e)) => {
                if let b"kwd-group" = e.name().as_ref() {
                    let mut group_type = String::new();
                    e.attributes().for_each(|attr| {
                        if let Ok(attr) = attr {
                            if attr.key.as_ref() == b"kwd-group-type" {
                                group_type = String::from_utf8_lossy(&attr.value).to_string();
                            }
                        };
                    });
                    let span = reader.read_text(e.name()).expect("err");
                    let just_string = remove_xml_tags(&span, " ");
                    //dbg!(&just_string);
                    txt.push([group_type + ":", just_string].join(" "));
                }
            }
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
    //let mut document_id = 0_usize;
    //let path_to_nxm_files = args.path_to_nxm_files + "/*.nxml";

    let embedding_data_file =
        BufReader::new(File::open("./test_doc/embedding.jsonl").expect("can open embedding.jsonl"));
    let mut doc_id_keywords = HashMap::<usize, String>::new();
    embedding_data_file.lines().for_each(|line| {
        let r: DocumentRecord =
            serde_json::from_str(line.unwrap().as_str()).expect("failed json conversion");
        let doc_id = r.document_id;

        doc_id_keywords.entry(doc_id).or_insert_with(|| {
            let path = PathBuf::from(args.path_to_nxm_files.clone() + "/" + r.file_name.as_str());
            let doc = get_keyword_groups(&path);
            let jsonl_record = serde_json::to_string(&(r.document_id, r.file_name, doc))
                .expect("json conversion fails");
            println!("{}", jsonl_record);
            jsonl_record
        });
    });
}
