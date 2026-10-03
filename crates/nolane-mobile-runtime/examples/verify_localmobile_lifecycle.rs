use nolane_mobile_runtime::{
    product::{
        lifecycle_tick_seconds,
        LocalMobileBundleManifest,
        LocalMobileProductRuntime,
        LEGACY_LOCAL_MOBILE_META_SCHEMA,
        LOCAL_MOBILE_META_SCHEMA,
    },
    read_persistent_mobile_state,
    write_persistent_mobile_state,
};
use serde_json::{json, Value};
use std::{env, fs, path::PathBuf};
use tempfile::tempdir;

fn main() {
    assert_eq!(lifecycle_tick_seconds("off"), 5);
    assert_eq!(lifecycle_tick_seconds("gentle"), 30);
    assert_eq!(lifecycle_tick_seconds("active"), 12);

    let mut args = env::args().skip(1);
    let bundle = PathBuf::from(
        args.next().expect("fixture root argument required"),
    );
    assert!(args.next().is_none(), "unexpected extra arguments");

    let manifest: LocalMobileBundleManifest = serde_json::from_slice(
        &fs::read(bundle.join("localmobile-manifest.json"))
            .expect("read LocalMobile manifest"),
    )
    .expect("parse LocalMobile manifest");

    let data = tempdir().expect("create lifecycle court data dir");
    let data_dir = data.path();
    let mut state = read_persistent_mobile_state(
        bundle.join("bootstrap-state.json"),
        Some(&manifest.source_checkpoint_sha256),
        None,
    )
    .expect("read bootstrap state");
    state.state.identity_id = "nolane-mobile-lifecycle-court".into();
    state.state.open_threads = vec![
        "finish mobile lifecycle parity".into(),
        "verify restart continuity".into(),
        "keep the product lightweight".into(),
    ];
    state.state.affect.concern = 1.0;
    state.state.relationship.closeness = 1.0;
    state.memories = vec![
        "User prefers concise answers about Nolane progress.".into(),
        "User prefers concise answers about Nolane progress.".into(),
        "Android LocalMobile must survive restart.".into(),
    ];
    write_persistent_mobile_state(
        data_dir.join("persistent-state.json"),
        &state,
        Some(&manifest.source_checkpoint_sha256),
        None,
    )
    .expect("write lifecycle state");

    fs::write(
        data_dir.join("local-mobile-meta.json"),
        serde_json::to_vec(&json!({
            "schema": LEGACY_LOCAL_MOBILE_META_SCHEMA,
            "state_version": 7,
            "memory_enabled": true,
            "initiative": "gentle",
        }))
        .unwrap(),
    )
    .unwrap();

    let mut host = LocalMobileProductRuntime::load_for_court(
        &bundle,
        data_dir,
    )
    .expect("load lifecycle host");

    let status = host.api("GET", "/v1/status", None).unwrap();
    assert_eq!(status["state_version"], 7);
    assert_eq!(
        status["lifecycle"]["schema"],
        LOCAL_MOBILE_META_SCHEMA,
    );
    assert_eq!(status["lifecycle"]["tick"], 0);

    host.api(
        "POST",
        "/v1/power",
        Some(json!({"enabled": true})),
    )
    .expect("power on");

    let chat = host
        .synthetic_court_chat("Hôm nay vui, tiếp tục Nolane nhé", 2)
        .expect("bounded lifecycle chat");
    assert_eq!(chat["state_version"], 8);

    let after_chat = host.api("GET", "/v1/status", None).unwrap();
    let last_user = after_chat["lifecycle"]["last_user_event_ms"]
        .as_u64()
        .expect("last user timestamp");
    assert!(
        after_chat["lifecycle"]["social_drive"].as_f64().unwrap() < 0.1
    );
    assert!(
        after_chat["lifecycle"]["curiosity"].as_f64().unwrap() > 0.35
    );

    let recent = host
        .synthetic_court_tick(last_user + 5 * 60 * 1000, 2)
        .expect("recent-user tick");
    assert_eq!(recent["state_version"], 9);
    assert_eq!(recent["initiative"]["speak"], false);
    assert_eq!(
        recent["initiative"]["reasons"][0],
        "user_recently_active",
    );
    assert_eq!(recent["rest"]["ran"], false);
    assert_eq!(recent["rest"]["reason"], "user_not_idle_enough");

    let idle = host
        .synthetic_court_tick(last_user + 31 * 60 * 1000, 2)
        .expect("idle lifecycle tick");
    assert_eq!(idle["state_version"], 10);
    assert_eq!(idle["rest"]["ran"], true);
    assert!(
        idle["rest"]["removed_duplicates"].as_u64().unwrap() >= 1
    );
    assert_eq!(idle["initiative"]["speak"], true);
    assert!(
        idle["initiative"]["score"].as_f64().unwrap() >= 0.66
    );
    assert!(
        idle["initiative"]["intent"]
            .as_str()
            .unwrap()
            .starts_with("follow_up:")
    );

    let before_restart = host.api("GET", "/v1/status", None).unwrap();
    let identity = before_restart["identity_id"]
        .as_str()
        .unwrap()
        .to_string();
    let lifecycle_before: Value =
        before_restart["lifecycle"].clone();
    drop(host);

    let mut reloaded = LocalMobileProductRuntime::load_for_court(
        &bundle,
        data_dir,
    )
    .expect("reload lifecycle host");
    let after_restart = reloaded.api("GET", "/v1/status", None).unwrap();
    assert_eq!(after_restart["identity_id"], identity);
    assert_eq!(after_restart["state_version"], 10);
    assert_eq!(
        after_restart["lifecycle"]["rest_cycles"],
        lifecycle_before["rest_cycles"],
    );
    assert_eq!(
        after_restart["lifecycle"]["last_user_event_ms"],
        lifecycle_before["last_user_event_ms"],
    );
    assert_eq!(
        after_restart["lifecycle"]["last_rest_ms"],
        lifecycle_before["last_rest_ms"],
    );

    reloaded
        .api(
            "POST",
            "/v1/power",
            Some(json!({"enabled": true})),
        )
        .unwrap();
    let cooldown = reloaded
        .synthetic_court_tick(last_user + 32 * 60 * 1000, 2)
        .expect("rest cooldown tick");
    assert_eq!(cooldown["rest"]["ran"], false);
    assert_eq!(cooldown["rest"]["reason"], "rest_cycle_cooldown");

    println!("NOLANE_V059_MOBILE_LIFECYCLE_PARITY_PASS");
}
