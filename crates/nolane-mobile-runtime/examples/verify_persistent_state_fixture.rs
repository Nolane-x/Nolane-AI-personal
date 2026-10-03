use nolane_mobile_runtime::{
    read_persistent_mobile_state,
    MobileRuntime,
};
use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct Fixture {
    source_checkpoint_sha256: String,
    prompt_contract_file_sha256: String,
}

fn main() {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(
        args.next().expect("fixture root argument required"),
    );
    assert!(args.next().is_none(), "unexpected extra arguments");

    let fixture: Fixture = serde_json::from_slice(
        &fs::read(root.join("fixture.json")).expect("read fixture"),
    )
    .expect("parse fixture");

    let state_path = root.join("persistent-state.json");
    let state = read_persistent_mobile_state(
        &state_path,
        Some(&fixture.source_checkpoint_sha256),
        Some(4),
    )
    .expect("read Python-authored persistent state");
    assert_eq!(state.profile.preferred_name, "Thuận");
    assert_eq!(state.state.identity_id, "identity-v055-mobile");
    assert_eq!(state.memories.len(), 2);

    let runtime = MobileRuntime::load_with_prompt_contract_and_persistent_state(
        root.join("package"),
        root.join("tokenizer.json"),
        root.join("tokenizer_config.json"),
        root.join("prompt-contract.json"),
        &fixture.prompt_contract_file_sha256,
        Some(&fixture.source_checkpoint_sha256),
        &state_path,
    )
    .expect("load native runtime with persistent state");

    let payload = runtime
        .persistent_product_payload(
            "reply",
            "conversation",
            Some("Tiếp tục dự án Nolane nhé"),
        )
        .expect("build product payload from persisted state");
    assert_eq!(payload.profile.preferred_name, "Thuận");
    assert_eq!(payload.state.identity_id, "identity-v055-mobile");
    assert_eq!(payload.memories.len(), 2);

    let generated = runtime
        .generate_persistent_product_seeded(
            "reply",
            "conversation",
            Some("Tiếp tục dự án Nolane nhé"),
            0x55C0FFEE,
        )
        .expect("generate from persistent mobile state");
    assert!(!generated.prompt_token_ids.is_empty());
    assert!(!generated.generated_token_ids.is_empty());

    let roundtrip = root.join("persistent-state-rust-roundtrip.json");
    runtime
        .save_persistent_state(&roundtrip)
        .expect("save persistent state from native runtime");
    let reloaded = read_persistent_mobile_state(
        &roundtrip,
        Some(&fixture.source_checkpoint_sha256),
        Some(4),
    )
    .expect("reload Rust-authored persistent state");
    assert_eq!(reloaded, state);

    println!("NOLANE_V055_PERSISTENT_MOBILE_STATE_BRIDGE_PASS");
}
