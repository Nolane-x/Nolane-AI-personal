use serde::Deserialize;
use std::{env, fs, path::PathBuf};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct Row {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    repo_id: String,
    revision: String,
    vocab_size_with_added: usize,
    rows: Vec<Row>,
}

fn main() {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().expect("fixture root argument required"));
    assert!(args.next().is_none(), "unexpected extra arguments");

    let fixture: Fixture = serde_json::from_slice(
        &fs::read(root.join("fixture.json")).expect("read fixture.json"),
    )
    .expect("parse fixture.json");
    assert_eq!(
        fixture.schema,
        "NOLANE-V051-REAL-QWEN-TOKENIZER-FIXTURE-V1"
    );
    assert_eq!(fixture.repo_id, "Qwen/Qwen3-1.7B");
    assert_eq!(
        fixture.revision,
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    );

    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
        .expect("load exact pinned Qwen tokenizer.json");
    assert_eq!(
        tokenizer.get_vocab_size(true),
        fixture.vocab_size_with_added
    );

    for (index, row) in fixture.rows.iter().enumerate() {
        let encoding = tokenizer
            .encode(row.text.as_str(), false)
            .unwrap_or_else(|error| panic!("encode row {index}: {error}"));
        assert_eq!(
            encoding.get_ids(),
            row.ids.as_slice(),
            "token ids mismatch at row {index}"
        );
        let decoded = tokenizer
            .decode(&row.ids, true)
            .unwrap_or_else(|error| panic!("decode row {index}: {error}"));
        assert_eq!(
            decoded,
            row.decoded,
            "decoded text mismatch at row {index}"
        );
    }

    println!("NOLANE_REAL_QWEN_TOKENIZER_RUST_PASS");
}
