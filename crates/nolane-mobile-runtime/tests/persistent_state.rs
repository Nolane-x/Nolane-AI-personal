use nolane_mobile_runtime::{
    MobileProfile,
    MobileStateStore,
    ProductPayloadState,
};
use std::fs;
use tempfile::tempdir;

fn checkpoint(byte: char) -> String {
    std::iter::repeat(byte).take(64).collect()
}

#[test]
fn mobile_state_initializes_desktop_compatible_product_defaults() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    let bundle = store
        .load_or_initialize(
            identity_id: "identity-1",
            checkpoint_sha256: &checkpoint('a'),
            latent_dim: 4,
        )
        .unwrap();

    assert_eq!(bundle.product.state.identity_id, "identity-1");
    assert_eq!(bundle.product.profile.preferred_name, "");
    assert_eq!(bundle.product.profile.language, "auto");
    assert_eq!(bundle.product.profile.response_length, "balanced");
    assert_eq!(bundle.product.profile.conversation_style, "natural");
    assert_eq!(bundle.product.profile.initiative, "gentle");
    assert!(bundle.product.profile.memory_enabled);
    assert_eq!(bundle.product.state.relationship.closeness, 0.05);
    assert_eq!(bundle.product.state.relationship.trust, 0.05);
    assert_eq!(bundle.product.state.relationship.familiarity, 0.0);
    assert_eq!(bundle.product.state.relationship.interaction_count, 0);
    assert_eq!(bundle.product.state.affect.valence, 0.0);
    assert_eq!(bundle.product.state.affect.energy, 0.65);
    assert_eq!(bundle.product.state.affect.playfulness, 0.45);
    assert_eq!(bundle.product.state.affect.concern, 0.0);
    assert_eq!(bundle.product.state.affect.irritation, 0.0);
    assert!(bundle.product.state.open_threads.is_empty());

    assert_eq!(bundle.latent.identity_id, "identity-1");
    assert_eq!(bundle.latent.checkpoint_sha256, checkpoint('a'));
    assert_eq!(bundle.latent.latent_dim, 4);
    assert_eq!(bundle.latent.values, vec![0.0; 4]);
    assert!(!bundle.latent_reinitialized);
    assert!(bundle.archived_latent.is_none());

    bundle.product.verify().unwrap();
    bundle.latent.verify().unwrap();
}

#[test]
fn profile_state_and_latent_updates_are_durable_and_versioned() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    let initial = store
        .load_or_initialize(
            identity_id: "identity-2",
            checkpoint_sha256: &checkpoint('b'),
            latent_dim: 3,
        )
        .unwrap();

    let profile = MobileProfile {
        preferred_name: "Tài".into(),
        language: "vi".into(),
        response_length: "compact".into(),
        conversation_style: "direct".into(),
        initiative: "active".into(),
        memory_enabled: false,
        personal_instruction: "Nói ngắn và cụ thể.".into(),
    };
    let updated_profile = store.update_profile(profile.clone()).unwrap();
    assert_eq!(updated_profile.version, initial.product.version + 1);
    assert_eq!(updated_profile.profile, profile);

    let mut state: ProductPayloadState = updated_profile.state.clone();
    state.relationship.closeness = 0.75;
    state.relationship.trust = 0.80;
    state.relationship.familiarity = 0.60;
    state.relationship.interaction_count = 42;
    state.affect.valence = 0.25;
    state.affect.energy = 0.72;
    state.affect.playfulness = 0.50;
    state.affect.concern = 0.10;
    state.affect.irritation = 0.05;
    state.open_threads = vec!["finish Android local runtime".into()];

    let updated_state = store.update_product_state(state.clone()).unwrap();
    assert_eq!(updated_state.version, updated_profile.version + 1);
    assert_eq!(updated_state.state.identity_id, "identity-2");
    assert_eq!(updated_state.state.open_threads, state.open_threads);

    let latent = store
        .update_latent(
            vec![0.1, -0.2, 0.3],
            source_state_version: updated_state.version,
        )
        .unwrap();
    assert_eq!(latent.sequence, 1);
    assert_eq!(latent.source_state_version, updated_state.version);

    let reopened = MobileStateStore::new(root.path())
        .load_or_initialize(
            identity_id: "identity-2",
            checkpoint_sha256: &checkpoint('b'),
            latent_dim: 3,
        )
        .unwrap();
    assert_eq!(reopened.product.profile, profile);
    assert_eq!(reopened.product.version, updated_state.version);
    assert_eq!(reopened.product.state.open_threads, state.open_threads);
    assert_eq!(reopened.latent.values, vec![0.1, -0.2, 0.3]);
    assert_eq!(reopened.latent.sequence, 1);
    assert!(!reopened.latent_reinitialized);
}

#[test]
fn checkpoint_change_archives_only_latent_and_preserves_product_state() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    store
        .load_or_initialize(
            identity_id: "identity-3",
            checkpoint_sha256: &checkpoint('c'),
            latent_dim: 2,
        )
        .unwrap();

    let profile = MobileProfile {
        preferred_name: "Persistent Name".into(),
        ..MobileProfile::default()
    };
    let product = store.update_profile(profile.clone()).unwrap();
    store
        .update_latent(
            vec![0.4, -0.6],
            source_state_version: product.version,
        )
        .unwrap();

    let migrated = store
        .load_or_initialize(
            identity_id: "identity-3",
            checkpoint_sha256: &checkpoint('d'),
            latent_dim: 2,
        )
        .unwrap();

    assert!(migrated.latent_reinitialized);
    let archive = migrated.archived_latent.as_ref().unwrap();
    assert!(archive.is_file());
    assert_eq!(migrated.product.profile, profile);
    assert_eq!(migrated.latent.checkpoint_sha256, checkpoint('d'));
    assert_eq!(migrated.latent.values, vec![0.0, 0.0]);

    let archived_text = fs::read_to_string(archive).unwrap();
    assert!(archived_text.contains(&checkpoint('c')));
    assert!(archived_text.contains("0.4"));
}

#[test]
fn identity_mismatch_never_silently_adopts_product_state() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    store
        .load_or_initialize(
            identity_id: "identity-a",
            checkpoint_sha256: &checkpoint('e'),
            latent_dim: 2,
        )
        .unwrap();

    let error = store
        .load_or_initialize(
            identity_id: "identity-b",
            checkpoint_sha256: &checkpoint('e'),
            latent_dim: 2,
        )
        .unwrap_err();
    assert!(error.to_string().contains("product identity mismatch"));
}

#[test]
fn tampered_main_state_fails_closed_even_when_valid_temp_exists() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    store
        .load_or_initialize(
            identity_id: "identity-4",
            checkpoint_sha256: &checkpoint('f'),
            latent_dim: 2,
        )
        .unwrap();

    let main = store.product_state_path();
    let temp = root.path().join(".product-state.json.tmp");
    fs::copy(&main, &temp).unwrap();

    let mut payload: serde_json::Value =
        serde_json::from_slice(&fs::read(&main).unwrap()).unwrap();
    payload["profile"]["preferred_name"] =
        serde_json::Value::String("tampered".into());
    fs::write(&main, serde_json::to_vec(&payload).unwrap()).unwrap();

    let error = store.load_product_state().unwrap_err();
    assert!(error.to_string().contains("digest mismatch"));
    assert!(main.is_file());
    assert!(temp.is_file());
}

#[test]
fn valid_temp_recovers_only_when_main_file_is_missing() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    let created = store
        .load_or_initialize(
            identity_id: "identity-5",
            checkpoint_sha256: &checkpoint('1'),
            latent_dim: 2,
        )
        .unwrap();

    let main = store.product_state_path();
    let temp = root.path().join(".product-state.json.tmp");
    fs::rename(&main, &temp).unwrap();
    assert!(!main.exists());
    assert!(temp.exists());

    let recovered = store.load_product_state().unwrap().unwrap();
    assert_eq!(recovered.digest, created.product.digest);
    assert!(main.is_file());
    assert!(!temp.exists());
}

#[test]
fn state_update_cannot_change_identity_and_invalid_latent_is_rejected() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    let initial = store
        .load_or_initialize(
            identity_id: "identity-6",
            checkpoint_sha256: &checkpoint('2'),
            latent_dim: 2,
        )
        .unwrap();

    let mut wrong_state = initial.product.state.clone();
    wrong_state.identity_id = "other".into();
    let error = store.update_product_state(wrong_state).unwrap_err();
    assert!(error
        .to_string()
        .contains("identity cannot change"));

    let error = store
        .update_latent(
            vec![f32::NAN, 0.0],
            source_state_version: 0,
        )
        .unwrap_err();
    assert!(error.to_string().contains("non-finite"));
}

#[test]
fn persisted_state_builds_the_existing_frozen_product_payload() {
    let root = tempdir().unwrap();
    let store = MobileStateStore::new(root.path());
    let bundle = store
        .load_or_initialize(
            identity_id: "identity-7",
            checkpoint_sha256: &checkpoint('3'),
            latent_dim: 2,
        )
        .unwrap();

    let payload = bundle
        .product
        .to_payload(
            mode: "reply",
            intent: "conversation",
            user_text: Some("xin chào".into()),
            memories: vec!["user likes concise answers".into()],
        )
        .unwrap();

    payload.validate().unwrap();
    assert_eq!(payload.schema, "NOLANE-V053-PRODUCT-PAYLOAD-INPUT-V1");
    assert_eq!(payload.profile.language, "auto");
    assert_eq!(payload.state.identity_id, "identity-7");
    assert_eq!(payload.user_text.as_deref(), Some("xin chào"));
    assert_eq!(payload.memories.len(), 1);
}
