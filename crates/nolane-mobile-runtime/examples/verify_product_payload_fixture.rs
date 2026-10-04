use nolane_mobile_runtime::{
    FrozenPromptContract,
    ProductPayloadInput,
    PRODUCT_SYSTEM_PROMPT,
};
use serde::Deserialize;
use std::{env, fs, path::PathBuf};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct ProductRow {
    name: String,
    payload: ProductPayloadInput,
    system_prompt: String,
    user_payload: String,
    rendered: String,
    ids: Vec<u32>,
    max_new_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ProductFixture {
    schema: String,
    repo_id: String,
    revision: String,
    prompt_contract_file_sha256: String,
    rows: Vec<ProductRow>,
}

fn main() {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(
        args.next().expect("fixture root argument required"),
    );
    assert!(args.next().is_none(), "unexpected extra arguments");

    let fixture: ProductFixture = serde_json::from_slice(
        &fs::read(root.join("product-payload-fixture.json"))
            .expect("read product payload fixture"),
    )
    .expect("parse product payload fixture");

    assert_eq!(
        fixture.schema,
        "NOLANE-V053-REAL-QWEN-PRODUCT-PAYLOAD-V1"
    );
    assert_eq!(fixture.repo_id, "Qwen/Qwen3-1.7B");
    assert_eq!(
        fixture.revision,
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    );
    assert!(!fixture.rows.is_empty());

    let contract = FrozenPromptContract::load(
        root.join("prompt-contract.json"),
        root.join("tokenizer.json"),
        root.join("tokenizer_config.json"),
        &fixture.prompt_contract_file_sha256,
    )
    .expect("load frozen product prompt contract");

    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
        .expect("load exact pinned Qwen tokenizer");

    for (index, row) in fixture.rows.iter().enumerate() {
        assert_eq!(
            row.system_prompt,
            PRODUCT_SYSTEM_PROMPT,
            "product system prompt mismatch at {} ({})",
            index,
            row.name,
        );

        let user_payload = row
            .payload
            .render_user_payload()
            .unwrap_or_else(|error| {
                panic!(
                    "render product payload row {} ({}): {}",
                    index,
                    row.name,
                    error
                )
            });
        assert_eq!(
            user_payload,
            row.user_payload,
            "product user payload mismatch at {} ({})",
            index,
            row.name,
        );

        let max_new_tokens = row
            .payload
            .max_new_tokens()
            .unwrap_or_else(|error| {
                panic!(
                    "resolve token budget row {} ({}): {}",
                    index,
                    row.name,
                    error
                )
            });
        assert_eq!(
            max_new_tokens,
            row.max_new_tokens,
            "product token budget mismatch at {} ({})",
            index,
            row.name,
        );

        let rendered = contract.render(
            PRODUCT_SYSTEM_PROMPT,
            &user_payload,
        );
        assert_eq!(
            rendered,
            row.rendered,
            "full product prompt mismatch at {} ({})",
            index,
            row.name,
        );

        let encoding = tokenizer
            .encode(rendered.as_str(), false)
            .unwrap_or_else(|error| {
                panic!(
                    "encode product prompt row {} ({}): {}",
                    index,
                    row.name,
                    error
                )
            });
        assert_eq!(
            encoding.get_ids(),
            row.ids.as_slice(),
            "product prompt token IDs mismatch at {} ({})",
            index,
            row.name,
        );
    }

    println!("NOLANE_V053_PRODUCT_PAYLOAD_RUST_PASS");
}
