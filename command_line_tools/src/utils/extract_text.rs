use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{Error, ErrorKind, Write};
use std::path::Path;

use lopdf::{Document, Object};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json;

static IGNORE: &[&str] = &[
    "Length",
    "BBox",
    "FormType",
    "Matrix",
    "Resources",
    "Type",
    "XObject",
    "Subtype",
    "Filter",
    "ColorSpace",
    "Width",
    "Height",
    "BitsPerComponent",
    "Length1",
    "Length2",
    "Length3",
    "PTEX.FileName",
    "PTEX.PageNumber",
    "PTEX.InfoDict",
    "FontDescriptor",
    "ExtGState",
    "Font",
    "MediaBox",
    "Annot",
];

#[derive(Debug, Deserialize, Serialize)]
pub struct PdfText {
    pub text: BTreeMap<u32, Vec<String>>, // Key is page number
    pub errors: Vec<String>,
}

pub fn filter_func(object_id: (u32, u16), object: &mut Object) -> Option<((u32, u16), Object)> {
    if IGNORE.contains(&object.type_name().unwrap_or_default()) {
        return None;
    }
    if let Ok(d) = object.as_dict_mut() {
        d.remove(b"Font");
        d.remove(b"Resources");
        d.remove(b"Producer");
        d.remove(b"ModDate");
        d.remove(b"Creator");
        d.remove(b"ProcSet");
        d.remove(b"XObject");
        d.remove(b"MediaBox");
        d.remove(b"Annots");
        if d.is_empty() {
            return None;
        }
    }
    Some((object_id, object.to_owned()))
}

pub fn load_pdf<P: AsRef<Path>>(path: P) -> Result<Document, Error> {
    Document::load_filtered(path, filter_func)
        .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))
}

pub fn get_pdf_text(doc: &Document) -> Result<PdfText, Error> {
    let mut pdf_text: PdfText = PdfText {
        text: BTreeMap::new(),
        errors: Vec::new(),
    };
    let pages: Vec<Result<(u32, Vec<String>), Error>> = doc
        .get_pages()
        .into_par_iter()
        .map(
            |(page_num, page_id): (u32, (u32, u16))| -> Result<(u32, Vec<String>), Error> {
                let text = doc.extract_text(&[page_num]).map_err(|e| {
                    Error::new(
                        ErrorKind::Other,
                        format!(
                            "Failed to extract text from page {page_num} id={page_id:?}: {e:?}"
                        ),
                    )
                })?;
                Ok((
                    page_num,
                    text.split('\n')
                        .map(|s| s.trim_end().to_string())
                        .collect::<Vec<String>>(),
                ))
            },
        )
        .collect();
    for page in pages {
        match page {
            Ok((page_num, lines)) => {
                pdf_text.text.insert(page_num, lines);
            }
            Err(e) => {
                pdf_text.errors.push(e.to_string());
            }
        }
    }
    Ok(pdf_text)
}

pub fn pdf2text<P: AsRef<Path> + Debug>(
    path: P,
    output: P,
    pretty: bool,
    password: &str,
) -> Result<(), Error> {
    println!("Load {path:?}");
    let mut doc = load_pdf(&path)?;
    if doc.is_encrypted() {
        doc.decrypt(password)
            .map_err(|_err| Error::new(ErrorKind::InvalidInput, "Failed to decrypt"))?;
    }
    let text = get_pdf_text(&doc)?;
    if !text.errors.is_empty() {
        eprintln!("{path:?} has {} errors:", text.errors.len());
        for error in &text.errors[..10] {
            eprintln!("{error:?}");
        }
    }
    let data = match pretty {
        true => serde_json::to_string_pretty(&text).unwrap(),
        false => serde_json::to_string(&text).unwrap(),
    };
    println!("Write {output:?}");
    let mut f = File::create(output)?;
    f.write_all(data.as_bytes())?;
    Ok(())
}
