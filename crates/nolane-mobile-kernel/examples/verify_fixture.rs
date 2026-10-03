use nolane_mobile_kernel::MobileKernel;
use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct GoldenStep {
    token_id: usize,
    logits: Vec<f32>,
    state: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    schema: String,
    source_checkpoint_sha256: String,
    latent: Vec<f32>,
    tokens: Vec<usize>,
    initial_state: Vec<f32>,
    steps: Vec<GoldenStep>,
    atol: f32,
    rtol: f32,
}

fn close(a: f32, b: f32, atol: f32, rtol: f32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn compare(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} length mismatch"
    );
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let error = (*a - *b).abs();
        if error > worst {
            worst = error;
            worst_index = index;
        }
        assert!(
            close(*a, *b, atol, rtol),
            "{label} mismatch at {index}: actual={a} expected={b} abs_error={error}"
        );
    }
    eprintln!("{label}: PASS len={} worst_abs={} at={}", actual.len(), worst, worst_index);
}

fn main() {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().expect("fixture root argument required"));
    assert!(args.next().is_none(), "unexpected extra arguments");

    let fixture: GoldenFixture = serde_json::from_slice(
        &fs::read(root.join("fixture.json")).expect("read fixture.json"),
    )
    .expect("parse fixture.json");
    assert_eq!(
        fixture.schema,
        "NOLANE-V050-MOBILE-RUST-GOLDEN-V1"
    );
    assert_eq!(fixture.tokens.len(), fixture.steps.len());

    let kernel = MobileKernel::load(
        root.join("package"),
        Some(&fixture.source_checkpoint_sha256),
    )
    .expect("load mobile package");

    let mut state = kernel.init_state(&fixture.latent).expect("init state");
    compare(
        "initial_state",
        &state,
        &fixture.initial_state,
        fixture.atol,
        fixture.rtol,
    );

    for (index, golden) in fixture.steps.iter().enumerate() {
        assert_eq!(golden.token_id, fixture.tokens[index]);
        let output = kernel
            .step(golden.token_id, &state, &fixture.latent)
            .expect("mobile token step");
        compare(
            &format!("step[{index}].logits"),
            &output.logits,
            &golden.logits,
            fixture.atol,
            fixture.rtol,
        );
        compare(
            &format!("step[{index}].state"),
            &output.state,
            &golden.state,
            fixture.atol,
            fixture.rtol,
        );
        state = output.state;
    }

    println!("NOLANE_MOBILE_RUST_GOLDEN_PASS");
}
