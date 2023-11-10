use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::{Error as E, Result};
use clap::Parser;
use llm_chain::traits::Embeddings;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::with_payload_selector::SelectorOptions;
use qdrant_client::qdrant::WithPayloadSelector;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};

use candle_core::{self, Device, Module, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::mistral::Config;
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

#[derive(Serialize, Deserialize, Debug)]
struct DocumentRecord {
    file_name: String,
    document_id: usize,
    section_id: usize,
    chunk_id: usize,
    text: String,
    embedding_vec: Vec<f32>,
}

#[derive(Parser, Debug)]
#[clap(name = "query_qdrant_db")]
//#[clap(author, version)]
//#[clap(about, long_about = None)]
struct CmdOptions {
    #[clap(long, short, default_value_t = 5)]
    topn: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CmdOptions::parse();

    let mut config = QdrantClientConfig::from_url("http://localhost:6334");
    config.set_timeout(Duration::new(50000, 0));
    let client = QdrantClient::new(Some(config))?;
    let collections_list = client.list_collections().await?;
    dbg!(collections_list);
    let collection_name = "NBK1116_chunks_mistral_7B";

    let mut fn_to_keywords = HashMap::<String, Vec<String>>::default();
    let keyword_file = BufReader::new(File::open("./test_doc/keywords.jsonl")?);
    keyword_file.lines().for_each(|line| {
        let (_doc_id, file_name, keywords): (usize, String, Vec<String>) =
            serde_json::from_str(line.unwrap().as_str()).expect("failed json conversion");
        fn_to_keywords.insert(file_name, keywords);
    });

    let api = Api::new()?;

    let repo = api.repo(Repo::with_revision(
        "lmz/candle-mistral".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let model_filename = repo.get("model-q4k.gguf")?;
    let config = Config::config_7b_v0_1(false);
    let vb = quantized_var_builder::VarBuilder::from_gguf(model_filename)?;
    let vb_m = vb.pp("model");
    let embed_tokens = Embedding::new(32000, 4096, vb_m.pp("embed_tokens"))?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    tokenizer.clear();

    let mut rl = DefaultEditor::new()?;
    let mut query_strings = Vec::new();
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                //println!("Line: {}", line);
                if !line.trim().is_empty() {
                    query_strings.push(line);
                } else {
                    if query_strings.is_empty() {
                        continue;
                    }
                    let query_str = query_strings.join("\n");
                    rl.add_history_entry(query_str.clone().as_str())?;
                    //let embeddings = llm_chain_openai::embeddings::Embeddings::default();
                    //let embedded_vecs = embeddings.embed_texts(vec![query_str]).await.unwrap();

                    //println!("{}", points[1120].text);
                    let tokens = tokenizer
                        .tokenizer()
                        .encode(query_str, true)
                        .map_err(E::msg)?
                        .get_ids()
                        .to_vec();

                    let input = Tensor::new(&tokens[..], &Device::Cpu)
                        .expect("convert to tensor error")
                        .unsqueeze(0)
                        .expect("unsqueeze error");
                    let emb = embed_tokens
                        .forward(&input)
                        .expect("can't get embedding vector");
                    let emb: Vec<f32> = emb.mean(1).unwrap().squeeze(0).unwrap().to_vec1().unwrap();

                    let search_result = client
                        .search_points(&SearchPoints {
                            collection_name: collection_name.into(),
                            vector: emb,
                            filter: None,
                            limit: args.topn,
                            with_vectors: None,
                            with_payload: Some(WithPayloadSelector {
                                selector_options: Some(SelectorOptions::Enable(true)),
                            }),
                            params: None,
                            score_threshold: None,
                            offset: None,
                            ..Default::default()
                        })
                        .await?;
                    //dbg!(search_result);
                    search_result.result.into_iter().for_each(|p| {
                        //let payload = serde_json::to_string(&p.payload).expect("json conversion fails");
                        let file_name = serde_json::to_string(p.payload.get("file_name").unwrap())
                            .expect("json conversion fails")
                            .trim_matches('"')
                            .to_string();
                        let document_id =
                            serde_json::to_string(p.payload.get("document_id").unwrap())
                                .expect("json conversion fails")
                                .trim_matches('"')
                                .parse::<usize>()
                                .expect("number parsing error");
                        let section_id =
                            serde_json::to_string(p.payload.get("section_id").unwrap())
                                .expect("json conversion fails")
                                .trim_matches('"')
                                .parse::<usize>()
                                .expect("number parsing error");
                        let chunk_id =
                            serde_json::to_string(p.payload.get("chunk_id").unwrap())
                                .expect("json conversion fails")
                                .trim_matches('"')
                                .parse::<usize>()
                                .expect("number parsing error");
                        let text =
                            serde_json::to_string(p.payload.get("text").unwrap())
                                .expect("json conversion fails");
                        let tmp = vec![];
                        let keywords = fn_to_keywords.get(&file_name).unwrap_or(&tmp);
                        let mut keywords = keywords.clone();
                        keywords.sort();
                        let prefix = file_name.as_str().split('.').next().unwrap();
                        let url = "https://www.ncbi.nlm.nih.gov/books/n/gene/".to_string() + prefix;
                        println!(

                            "+++++++++++++++++++\nscore: {}\ndocument name: {}\nURL: {}\ndocuemnt id: {}:{}:{}\n{}\n{}\n===================\n",
                            p.score,
                            file_name,
                            url,
                            document_id,
                            section_id,
                            chunk_id,
                            keywords.join("\n"),
                            text
                        );
                    });
                    query_strings.clear();
                }
            }
            Err(ReadlineError::Interrupted) => break,
            Err(ReadlineError::Eof) => break,
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}
