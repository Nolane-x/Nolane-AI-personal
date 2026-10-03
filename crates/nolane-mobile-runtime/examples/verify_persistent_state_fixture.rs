use nolane_mobile_runtime::MobileStateStore;
use serde::Deserialize;
use serde_json::Value;
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    identity_id: String,
    checkpoint_sha256: String,
    latent_dim: usize,
    full_profile: Value,
    payload: Value,
    latent_values: Vec<f32>,
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
    assert_eq!(
        fixture.schema,
        "NOLANE-V055-MOBILE-STATE-PARITY-FIXTURE-V1"
    );

    let state_root = root.join("native-state");
    if state_root.exists() {
        fs::remove_dir_all(&state_root).expect("clear native state root");
    }
    let store = MobileStateStore::new(&state_root);
    let bundle = store
        .load_or_initialize(
            &fixture.identity_id,
            &fixture.checkpoint_sha256,
            fixture.latent_dim,
        )
        .expect("initialize native state");

    assert_eq!(
        serde_json::to_value(&bundle.product.profile).unwrap(),
        fixture.full_profile,
        "mobile profile defaults drifted from desktop ProductProfile",
    );
    assert_eq!(
        bundle.latent.values,
        fixture.latent_values,
        "mobile latent default drifted from desktop clean-latent policy",
    );
    assert_eq!(
        bundle.latent.identity_id,
        fixture.identity_id,
    );
    assert_eq!(
        bundle.latent.checkpoint_sha256,
        fixture.checkpoint_sha256,
    );

    let payload = bundle
        .product
        .to_payload(
            "reply",
            "conversation",
            Some("xin chào".into()),
            Vec::new(),
        )
        .expect("build native product payload");
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        fixture.payload,
        "native persistent state did not reproduce desktop product payload",
    );

    println!("NOLANE_V055_MOBILE_STATE_PARITY_PASS");
}
