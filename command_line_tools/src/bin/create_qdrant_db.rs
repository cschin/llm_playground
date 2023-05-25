use std::fs::File;
use std::io::{BufReader, BufRead};
use std::time::Duration;

use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, VectorParams, VectorsConfig};
use serde::{Deserialize, Serialize};
use qdrant_client::client::Payload;

#[derive(Serialize, Deserialize, Debug)]
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
    let collection_name = "NBK1116";
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
    
    let points = embedding_data_file.lines().map(|line| {
        let r: DocumentRecord = serde_json::from_str(line.unwrap().as_str()).expect("failed json conversion");
        let mut payload = Payload::new();
        payload.insert("file_name", r.file_name);
        payload.insert("document_id", r.document_id.to_string());
        payload.insert("section_id", r.section_id.to_string());
        payload.insert("chunk_id", r.chunk_id.to_string());
        payload.insert("text", r.text);
        let id = r.document_id << 32 |  r.section_id << 16 | r.section_id; 
        PointStruct::new(id as u64, r.embedding_vec, payload)
    }).collect::<Vec<_>>();
    client
        .upsert_points_blocking(collection_name, points, None)
        .await?;
    println!("data stored in the vector store db");

    Ok(())
}
