use nolane_mobile_runtime::MobileRuntime;
use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    source_checkpoint_sha256: String,
    latent: Vec<f32>,
    prompt: String,
    system_prompt: String,
    user_prompt: String,
    prompt_contract_file_sha256: String,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    stopped_on_eos: bool,
    generation_final_state: Vec<f32>,
    atol: f32,
    rtol: f32,
}

fn close(a: f32, b: f32, atol: f32, rtol: f32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn main() {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().expect("fixture root argument required"));
    assert!(args.next().is_none(), "unexpected extra arguments");

    let fixture: Fixture = serde_json::from_slice(
        &fs::read(root.join("fixture.json")).expect("read fixture"),
    )
    .expect("parse fixture");
    assert_eq!(fixture.schema, "NOLANE-V050-MOBILE-RUST-GOLDEN-V1");

    let runtime = MobileRuntime::load_with_prompt_contract(
        root.join("package"),
        root.join("tokenizer.json"),
        root.join("tokenizer_config.json"),
        root.join("prompt-contract.json"),
        &fixture.prompt_contract_file_sha256,
        Some(&fixture.source_checkpoint_sha256),
        fixture.latent.clone(),
    )
    .expect("load prompt-bound native mobile runtime");

    let rendered = runtime
        .render_chat_prompt(&fixture.system_prompt, &fixture.user_prompt)
        .expect("render native chat prompt");
    assert_eq!(rendered, fixture.prompt);

    let generated = runtime
        .generate_chat_greedy(
            &fixture.system_prompt,
            &fixture.user_prompt,
            fixture.max_new_tokens,
        )
        .expect("generate from frozen chat prompt");

    assert_eq!(generated.prompt_token_ids, fixture.prompt_token_ids);
    assert_eq!(generated.generated_token_ids, fixture.generated_token_ids);
    assert_eq!(generated.text, fixture.generated_text);
    assert_eq!(generated.stopped_on_eos, fixture.stopped_on_eos);
    assert_eq!(
        generated.final_state.len(),
        fixture.generation_final_state.len()
    );
    for (index, (actual, expected)) in generated
        .final_state
        .iter()
        .zip(fixture.generation_final_state.iter())
        .enumerate()
    {
        assert!(
            close(*actual, *expected, fixture.atol, fixture.rtol),
            "chat final state mismatch at {index}: actual={actual} expected={expected}"
        );
    }

    assert_eq!(
        runtime.prompt_contract_file_sha256(),
        Some(fixture.prompt_contract_file_sha256.as_str())
    );
    println!("NOLANE_MOBILE_CHAT_PROMPT_GENERATION_PASS");
}
