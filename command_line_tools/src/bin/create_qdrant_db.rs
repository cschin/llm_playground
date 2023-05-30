use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::Result;
use qdrant_client::client::Payload;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, VectorParams, VectorsConfig};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DocumentRecord {
    file_name: String,
    document_id: usize,
    section_id: usize,
    chunk_id: usize,
    text: String,
    embedding_vec: Vec<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut config = QdrantClientConfig::from_url("http://localhost:6334");
    config.set_timeout(Duration::new(50000, 0));
    let client = QdrantClient::new(Some(config))?;
    let collections_list = client.list_collections().await?;
    dbg!(collections_list);
    let collection_name = "NBK1116_chunks";
    client.delete_collection(collection_name).await?;
    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 1536,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: Some(false),
                })),
            }),
            ..Default::default()
        })
        .await?;

    let embedding_data_file =
        BufReader::new(File::open("./test_doc/embedding.jsonl").expect("can open embedding.jsonl"));
    let mut all_doc_records = Vec::<DocumentRecord>::new();
    let points = embedding_data_file
        .lines()
        .map(|line| {
            let r: DocumentRecord =
                serde_json::from_str(line.unwrap().as_str()).expect("failed json conversion");
            let r_keep = r.clone();
            all_doc_records.push(r_keep);

            let mut payload = Payload::new();
            payload.insert("file_name", r.file_name);
            payload.insert("document_id", r.document_id.to_string());
            payload.insert("section_id", r.section_id.to_string());
            payload.insert("chunk_id", r.chunk_id.to_string());
            payload.insert("text", r.text);
            let id = r.document_id << 32 | r.section_id << 16 | r.section_id;
            PointStruct::new(id as u64, r.embedding_vec, payload)
        })
        .collect::<Vec<_>>();

    client
        .upsert_points_blocking(collection_name, points, None)
        .await?;
    println!("data stored in the vector store db");

    // Document / Section level points
    let mut sec_to_records = HashMap::<(String, usize, usize), Vec<&DocumentRecord>>::new();
    let mut doc_to_records = HashMap::<(String, usize), Vec<&DocumentRecord>>::new();

    all_doc_records.iter().for_each(|r| {
        let file_name = r.file_name.clone();
        let doc_id = r.document_id;
        let sec_id = r.section_id;

        let e = sec_to_records
            .entry((file_name.clone(), doc_id, sec_id))
            .or_insert_with(Vec::new);
        e.push(r);
        let e = doc_to_records.entry((file_name, doc_id)).or_insert_with(Vec::new);
        e.push(r);
    });

    let collection_name = "NBK1116_sections";
    client.delete_collection(collection_name).await?;
    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 1536,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: Some(false),
                })),
            }),
            ..Default::default()
        })
        .await?;

    let points = sec_to_records.iter().map(|((file_name, doc_id, sec_id), records)| {
        let mut mean_vec: Vec<f32> = 
        records.iter().fold(Vec::new(), |mut m, &v| {
            if m.is_empty() {
                m = v.embedding_vec.clone();
                m
            } else {
                v.embedding_vec.iter().enumerate().for_each(|(i, v)| {m[i] += v;});
                m
            }
        }); 
        mean_vec.iter_mut().for_each(|v| {*v /= records.len() as f32;});
        
        let mut payload = Payload::new();

        payload.insert("file_name", file_name.clone());
        payload.insert("document_id", doc_id.to_string());
        payload.insert("section_id", sec_id.to_string());
        let id = doc_id << 32 | sec_id << 16;
        PointStruct::new(id as u64, mean_vec, payload)
    }).collect::<Vec<_>>();
        
    client
        .upsert_points_blocking(collection_name, points, None)
        .await?;
    println!("data stored in the vector store db");

    let collection_name = "NBK1116_documents";
    client.delete_collection(collection_name).await?;
    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 1536,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: Some(false),
                })),
            }),
            ..Default::default()
        })
        .await?;

    let points = doc_to_records.iter().map(|((file_name, doc_id), records)| {
        let mut mean_vec: Vec<f32> = 
        records.iter().fold(Vec::new(), |mut m, &v| {
            if m.is_empty() {
                m = v.embedding_vec.clone();
                m
            } else {
                v.embedding_vec.iter().enumerate().for_each(|(i, v)| {m[i] += v;});
                m
            }
        }); 
        mean_vec.iter_mut().for_each(|v| {*v /= records.len() as f32;});
        
        let mut payload = Payload::new();

        payload.insert("file_name", file_name.clone());
        payload.insert("document_id", doc_id.to_string());
        let id = doc_id << 32;
        PointStruct::new(id as u64, mean_vec, payload)
    }).collect::<Vec<_>>();
        
    client
        .upsert_points_blocking(collection_name, points, None)
        .await?;
    println!("data stored in the vector store db");

    Ok(())
}
