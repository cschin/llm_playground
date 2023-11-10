#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use glob::glob;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde::{Deserialize, Serialize};

use candle_transformers::models::mistral::Config;
use candle_core::{self, Device, Module, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::quantized_var_builder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(
        d1: usize,
        d2: usize,
        vb: quantized_var_builder::VarBuilder,
    ) -> candle_core::Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, d2);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Parser, Debug)]
#[clap(name = "get_mistral_7B_embed_vec")]
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
    let text = String::from_utf8_lossy(&buf[..]);
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
                    let just_string = remove_xml_tags(&span, " ");
                    sec_txt.push([String::from("reference: "), just_string].join(""));
                }
                _ => (),
            },
            Ok(Event::End(e)) => {
                if b"sec" == e.name().as_ref() {
                    txt.push(sec_txt.join("\n"));
                    //txt.push(String::from("\n\n"));
                    sec_txt.clear();
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

fn main() -> Result<()> {
    //CmdOptions::command().version(VERSION_STRING).get_matches();
    let args = CmdOptions::parse();

    // println!(
    //     "avx: {}, neon: {}, simd128: {}, f16c: {}",
    //     utils::with_avx(),
    //     utils::with_neon(),
    //     utils::with_simd128(),
    //     utils::with_f16c()
    // );

    let api = Api::new()?;

    let repo = api.repo(Repo::with_revision(
        "lmz/candle-mistral".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let model_filename = repo.get("model-q4k.gguf")?;
    let mut document_id_tuple: Vec<(String, usize)> = Vec::new();
    let mut document_id = 0_usize;
    let path_to_nxm_files = args.path_to_nxm_files + "/*.nxml";
    let config = Config::config_7b_v0_1(false);
    let vb = quantized_var_builder::VarBuilder::from_gguf(model_filename)?;
    let vb_m = vb.pp("model");
    // let model = QMistral::new(&config, vb)?;

    let embed_tokens = Embedding::new(32000, 4096, vb_m.pp("embed_tokens"))?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    tokenizer.clear();

    // let bpe = p50k_base().unwrap();

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
            .filter(|(_section_id, section)| !section.is_empty())
            .try_for_each(|(section_id, section)| -> Result<()> {
                let max_tokens_per_chunk = 512;
                let chunk_overlap = 64;

                let tokens = tokenizer
                    .tokenizer()
                    .encode(section, true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();

                let token_strings = tokens
                    .iter()
                    .flat_map(|t| tokenizer.next_token(*t))
                    .flatten()
                    .collect::<Vec<_>>();

                let chunks = (0..token_strings.len())
                    .step_by(max_tokens_per_chunk - chunk_overlap)
                    .map(|start| {
                        let end = if start + max_tokens_per_chunk > token_strings.len() {
                            token_strings.len()
                        } else {
                            start + max_tokens_per_chunk
                        };
                        let input = Tensor::new(&tokens[start..end], &Device::Cpu)
                            .expect("convert to tensor error")
                            .unsqueeze(0)
                            .expect("unsqueeze error");
                        let emb = embed_tokens
                            .forward(&input)
                            .expect("can't get embedding vector");
                        let emb: Vec<f32> =
                            emb.mean(1).unwrap().squeeze(0).unwrap().to_vec1().unwrap();

                        (token_strings[start..end].join(""), emb)
                    })
                    .enumerate()
                    .map(|(chunk_id, (chunk, emb))| {
                        (
                            file_name.clone(),
                            document_id,
                            section_id,
                            chunk_id,
                            chunk,
                            emb,
                        )
                    })
                    .collect::<Vec<_>>();

                // println!("{} {:?}", chunks.len(), chunks);
                all_chunk.extend(chunks);
                Ok(())
            })?;

        all_chunk
            .into_iter()
            .map(
                |(file_name, document_id, section_id, chunk_id, text, embedding_vec)| {
                    let r = DocumentRecord {
                        file_name,
                        document_id,
                        section_id,
                        chunk_id,
                        text,
                        embedding_vec,
                    };
                    serde_json::to_string(&r).expect("json conversion fails")
                },
            )
            .for_each(|r| println!("{}", r));
        document_id += 1;
    }
    Ok(())
}
