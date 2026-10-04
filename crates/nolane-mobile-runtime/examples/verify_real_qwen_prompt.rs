use nolane_mobile_runtime::FrozenPromptContract;
use serde::Deserialize;
use std::{env, fs, path::PathBuf};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct PromptRow {
    system: String,
    user: String,
    rendered: String,
    ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct PromptFixture {
    schema: String,
    repo_id: String,
    revision: String,
    prompt_contract_file_sha256: String,
    rows: Vec<PromptRow>,
}

fn main() {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().expect("fixture root argument required"));
    assert!(args.next().is_none(), "unexpected extra arguments");

    let fixture: PromptFixture = serde_json::from_slice(
        &fs::read(root.join("prompt-fixture.json"))
            .expect("read prompt-fixture.json"),
    )
    .expect("parse prompt fixture");
    assert_eq!(
        fixture.schema,
        "NOLANE-V052-REAL-QWEN-PROMPT-FIXTURE-V1"
    );
    assert_eq!(fixture.repo_id, "Qwen/Qwen3-1.7B");
    assert_eq!(
        fixture.revision,
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    );

    let contract = FrozenPromptContract::load(
        root.join("prompt-contract.json"),
        root.join("tokenizer.json"),
        root.join("tokenizer_config.json"),
        &fixture.prompt_contract_file_sha256,
    )
    .expect("load integrity-bound prompt contract");

    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
        .expect("load exact pinned Qwen tokenizer");

    for (index, row) in fixture.rows.iter().enumerate() {
        let rendered = contract.render(&row.system, &row.user);
        assert_eq!(
            rendered,
            row.rendered,
            "rendered prompt mismatch at row {index}"
        );

        let encoding = tokenizer
            .encode(rendered.as_str(), false)
            .unwrap_or_else(|error| panic!("encode prompt row {index}: {error}"));
        assert_eq!(
            encoding.get_ids(),
            row.ids.as_slice(),
            "rendered prompt token IDs mismatch at row {index}"
        );
    }

    println!("NOLANE_REAL_QWEN_PROMPT_CONTRACT_RUST_PASS");
}
