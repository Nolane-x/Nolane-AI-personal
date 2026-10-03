use nolane_mobile_runtime::{
    memory::read_mobile_memory_store,
    product::LocalMobileProductRuntime,
};
use serde_json::json;
use std::{env, path::PathBuf};
use tempfile::tempdir;

fn main() {
    let mut args = env::args().skip(1);
    let bundle = PathBuf::from(
        args.next().expect("fixture root argument required"),
    );
    assert!(args.next().is_none(), "unexpected extra arguments");

    let data = tempdir().expect("create v0.60 memory data dir");
    let memory_path = data.path().join("local-mobile-memory.json");
    let mut host = LocalMobileProductRuntime::load_for_court(
        &bundle,
        data.path(),
    )
    .expect("load LocalMobile for provenance memory court");

    let migrated = read_mobile_memory_store(&memory_path)
        .expect("read migrated memory store");
    assert_eq!(migrated.records.len(), 2);
    assert!(migrated.records.iter().all(|record| {
        record
            .metadata
            .get("migrated_from_v055_projection")
            .and_then(|value| value.as_bool())
            == Some(true)
    }));

    host.api(
        "POST",
        "/v1/power",
        Some(json!({"enabled": true})),
    )
    .expect("power on");

    host.synthetic_court_chat(
        "Tôi thích màu cam chủ đạo cho Nolane.",
        4,
    )
    .expect("first episodic chat");
    host.synthetic_court_chat(
        "Tôi rất thích màu cam chủ đạo cho Nolane.",
        4,
    )
    .expect("second episodic chat");

    let after_chat = read_mobile_memory_store(&memory_path)
        .expect("read post-chat memory store");
    assert!(after_chat.records.len() >= 5);
    let episodic = after_chat
        .records
        .iter()
        .filter(|record| {
            record.source_event_id
                .as_deref()
                .is_some_and(|value| value.starts_with("local-mobile-u-"))
        })
        .collect::<Vec<_>>();
    assert_eq!(episodic.len(), 2);
    assert!(episodic.iter().all(|record| {
        record.kind == "episodic"
            && record.confidence == 1.0
            && record.created_at_ms > 0
    }));

    let status = host
        .api("GET", "/v1/status", None)
        .expect("status after chats");
    assert_eq!(
        status["memory"]["records"].as_u64().unwrap() as usize,
        after_chat.records.len()
    );
    assert_eq!(status["memory"]["links"], 0);
    let observer_memories = after_chat
        .records
        .iter()
        .filter(|record| {
            record
                .metadata
                .get("proposed_by_observer")
                .and_then(|value| value.as_bool())
                == Some(true)
        })
        .collect::<Vec<_>>();
    assert_eq!(observer_memories.len(), 1);
    assert_eq!(observer_memories[0].kind, "preference");
    let last_user = status["lifecycle"]["last_user_event_ms"]
        .as_u64()
        .expect("last user timestamp");

    let rest = host
        .synthetic_court_tick(
            last_user + 31 * 60 * 1000,
            4,
        )
        .expect("run provenance REST court");
    assert_eq!(rest["rest"]["ran"], true);
    assert_eq!(
        rest["rest"]["provenance_rich_consolidation"],
        true
    );
    assert!(
        rest["rest"]["new_memories"]
            .as_u64()
            .expect("new memory count")
            >= 1
    );

    let after_rest = read_mobile_memory_store(&memory_path)
        .expect("read post-rest memory store");
    assert!(after_rest.records.len() > after_chat.records.len());
    assert!(after_rest.links.len() >= 2);
    assert!(after_rest.records.iter().any(|record| {
        record
            .metadata
            .get("rest_consolidated")
            .and_then(|value| value.as_bool())
            == Some(true)
            && record
                .metadata
                .get("source_memory_ids")
                .and_then(|value| value.as_array())
                .is_some_and(|ids| ids.len() >= 2)
    }));

    let count_before_memory_off = after_rest.records.len();
    host.api(
        "PUT",
        "/v1/profile",
        Some(json!({"memory_enabled": false})),
    )
    .expect("disable memory");
    host.synthetic_court_chat(
        "Tin nhắn này không được ghi thành memory.",
        4,
    )
    .expect("memory-disabled chat still works");
    let disabled = read_mobile_memory_store(&memory_path)
        .expect("read memory-disabled store");
    assert_eq!(disabled.records.len(), count_before_memory_off);

    drop(host);

    let mut reloaded = LocalMobileProductRuntime::load_for_court(
        &bundle,
        data.path(),
    )
    .expect("reload provenance memory host");
    let status = reloaded
        .api("GET", "/v1/status", None)
        .expect("reloaded status");
    assert_eq!(
        status["memory"]["records"].as_u64().unwrap() as usize,
        count_before_memory_off
    );
    assert!(
        status["memory"]["links"].as_u64().unwrap() >= 2
    );
    assert_eq!(status["memory_enabled"], false);

    let final_store = read_mobile_memory_store(&memory_path)
        .expect("final memory store");
    assert_eq!(final_store, disabled);

    println!("NOLANE_V060_PROVENANCE_MEMORY_RESTART_PASS");
}
