use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::Result;
use llm_chain::traits::Embeddings;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::with_payload_selector::SelectorOptions;
use qdrant_client::qdrant::WithPayloadSelector;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentRecord {
    pub score: f32,
    pub file_name: Option<String>,
    pub url: Option<String>,
    pub document_id: Option<usize>,
    pub section_id: Option<usize>,
    pub chunk_id: Option<usize>,
    pub keywords: Option<String>,
    pub text: Option<String>,
}

pub async fn query_doc_vec_db(text: &String, topn: u64) -> Result<Vec<DocumentRecord>> {
    let mut config = QdrantClientConfig::from_url("http://localhost:6334");
    config.set_timeout(Duration::new(50000, 0));
    let client = QdrantClient::new(Some(config))?;
    let collections_list = client.list_collections().await?;
    dbg!(collections_list);
    let collection_name = "NBK1116";

    let mut fn_to_keywords = HashMap::<String, Vec<String>>::default();
    let keyword_file = BufReader::new(File::open("../test_doc/keywords.jsonl")?);
    keyword_file.lines().into_iter().for_each(|line| {
        let (_doc_id, file_name, keywords): (usize, String, Vec<String>) =
            serde_json::from_str(line.unwrap().as_str()).expect("failed json conversion");
        fn_to_keywords.insert(file_name, keywords);
    });

    let query_str = text.clone();
    println!("{}", query_str.clone());
    let embeddings = llm_chain_openai::embeddings::Embeddings::default();
    let embedded_vecs = embeddings.embed_texts(vec![query_str.clone()]).await.unwrap();

    //println!("{}", points[1120].text);
    let search_result = client
        .search_points(&SearchPoints {
            collection_name: collection_name.into(),
            vector: embedded_vecs[0].clone(),
            filter: None,
            limit: topn,
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
    let mut return_docs = Vec::new();

    search_result.result.into_iter().for_each(|p| {

        let file_name = serde_json::to_string(p.payload.get("file_name").unwrap())
            .expect("json conversion fails")
            .trim_matches('"')
            .to_string();

        let document_id = serde_json::to_string(p.payload.get("document_id").unwrap())
            .expect("json conversion fails")
            .trim_matches('"')
            .parse::<usize>()
            .expect("number parsing error");

        let section_id = serde_json::to_string(p.payload.get("section_id").unwrap())
            .expect("json conversion fails")
            .trim_matches('"')
            .parse::<usize>()
            .expect("number parsing error");

        let chunk_id = serde_json::to_string(p.payload.get("chunk_id").unwrap())
            .expect("json conversion fails")
            .trim_matches('"')
            .parse::<usize>()
            .expect("number parsing error");

        let text =
            serde_json::to_string(p.payload.get("text").unwrap()).expect("json conversion fails");

        let tmp = vec![];
        let keywords = fn_to_keywords.get(&file_name).unwrap_or(&tmp);
        let mut keywords = keywords.clone();
        keywords.sort();
        let prefix = file_name.as_str().split(".").next().unwrap();
        let url = "https://www.ncbi.nlm.nih.gov/books/n/gene/".to_string() + prefix;
        let return_doc = DocumentRecord {
            score: p.score,
            file_name: Some(file_name),
            url: Some(url),
            document_id: Some(document_id),
            section_id: Some(section_id),
            chunk_id: Some(chunk_id),
            keywords: Some(keywords.join("\n")),
            text: Some(text),
        };

        return_docs.push(return_doc);
    });

    Ok(return_docs)
}
