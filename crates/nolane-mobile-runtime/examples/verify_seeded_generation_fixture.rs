use nolane_mobile_runtime::{
    MobileRuntime,
    SEEDED_SAMPLER_SCHEMA,
};
use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    source_checkpoint_sha256: String,
    latent: Vec<f32>,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    seeded_sampler_schema: String,
    sampling_seed: u64,
    sampling_temperature: f64,
    sampling_top_p: f64,
    sampled_max_new_tokens: usize,
    sampled_generated_token_ids: Vec<u32>,
    sampled_generated_text: String,
    sampled_stopped_on_eos: bool,
    sampled_final_state: Vec<f32>,
    atol: f32,
    rtol: f32,
}

fn close(a: f32, b: f32, atol: f32, rtol: f32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
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
    assert_eq!(fixture.schema, "NOLANE-V050-MOBILE-RUST-GOLDEN-V1");
    assert_eq!(fixture.seeded_sampler_schema, SEEDED_SAMPLER_SCHEMA);

    let runtime = MobileRuntime::load(
        root.join("package"),
        root.join("tokenizer.json"),
        Some(&fixture.source_checkpoint_sha256),
        fixture.latent.clone(),
    )
    .expect("load native mobile runtime");

    let generated = runtime
        .generate_seeded(
            &fixture.prompt,
            fixture.sampled_max_new_tokens,
            fixture.sampling_seed,
            fixture.sampling_temperature,
            fixture.sampling_top_p,
        )
        .expect("native seeded generation");

    assert_eq!(generated.prompt_token_ids, fixture.prompt_token_ids);
    assert_eq!(
        generated.generated_token_ids,
        fixture.sampled_generated_token_ids
    );
    assert_eq!(generated.text, fixture.sampled_generated_text);
    assert_eq!(
        generated.stopped_on_eos,
        fixture.sampled_stopped_on_eos
    );

    assert_eq!(
        generated.final_state.len(),
        fixture.sampled_final_state.len()
    );
    for (index, (actual, expected)) in generated
        .final_state
        .iter()
        .zip(fixture.sampled_final_state.iter())
        .enumerate()
    {
        assert!(
            close(*actual, *expected, fixture.atol, fixture.rtol),
            "seeded final state mismatch at {index}: actual={actual} expected={expected}"
        );
    }

    println!("NOLANE_V054_SEEDED_SAMPLING_PARITY_PASS");
}
