use nolane_mobile_runtime::product::LocalMobileProductRuntime;
use serde_json::json;
use std::{env, path::PathBuf};
use tempfile::tempdir;

fn main() {
    let mut args = env::args().skip(1);
    let bundle = PathBuf::from(
        args.next().expect("fixture root argument required"),
    );
    assert!(args.next().is_none(), "unexpected extra arguments");

    let data = tempdir().expect("create LocalMobile data directory");
    let strict = LocalMobileProductRuntime::load(
        &bundle,
        data.path(),
    );
    assert!(
        strict.is_err(),
        "court-only LocalMobile bundle must never pass strict release loading"
    );

    let mut host = LocalMobileProductRuntime::load_for_court(
        &bundle,
        data.path(),
    )
    .expect("load LocalMobile product host for court");

    let status = host
        .api("GET", "/v1/status", None)
        .expect("status");
    assert_eq!(status["phase"], "off");
    assert_eq!(status["powered"], false);
    assert_eq!(status["device"], "android-local-rust");
    assert!(
        status["identity_id"]
            .as_str()
            .expect("identity string")
            .starts_with("nolane-mobile-")
    );
    assert_eq!(status["interactions"], 17);

    let profile = host
        .api("GET", "/v1/profile", None)
        .expect("profile");
    assert_eq!(profile["preferred_name"], "Thuận");
    assert_eq!(profile["language"], "vi");
    assert_eq!(profile["initiative"], "gentle");
    assert_eq!(profile["memory_enabled"], true);

    let on = host
        .api(
            "POST",
            "/v1/power",
            Some(json!({"enabled": true})),
        )
        .expect("power on");
    assert_eq!(on["phase"], "on");

    let reply = host
        .api(
            "POST",
            "/v1/chat",
            Some(json!({"text": "Tiếp tục Nolane nhé"})),
        )
        .expect("native LocalMobile chat");
    assert_eq!(reply["state_version"], 1);

    let history = host
        .api("GET", "/v1/history", None)
        .expect("history");
    assert_eq!(
        history["messages"].as_array().expect("message array").len(),
        2
    );
    assert_eq!(history["messages"][0]["role"], "user");
    assert_eq!(history["messages"][0]["text"], "Tiếp tục Nolane nhé");
    assert_eq!(history["messages"][1]["role"], "assistant");

    let memory_status = host
        .api("GET", "/v1/status", None)
        .expect("memory status");
    let memories = memory_status["mind"]["memories"]
        .as_array()
        .expect("memory projection");
    assert_eq!(memories.len(), 2);
    assert_eq!(memories[0]["kept"], false);

    host.api(
        "POST",
        "/v1/memory",
        Some(json!({
            "action": "keep",
            "memory_id": "slot:0",
        })),
    )
    .expect("keep LocalMobile memory");
    let kept = host.api("GET", "/v1/status", None).unwrap();
    assert_eq!(kept["mind"]["memories"][0]["kept"], true);

    host.api(
        "POST",
        "/v1/memory",
        Some(json!({
            "action": "edit",
            "memory_id": "slot:0",
            "text": "Người dùng đang xây Nolane v1.",
        })),
    )
    .expect("edit LocalMobile memory");
    let edited = host.api("GET", "/v1/status", None).unwrap();
    assert_eq!(
        edited["mind"]["memories"][0]["text"],
        "Người dùng đang xây Nolane v1."
    );
    assert_eq!(edited["mind"]["memories"][0]["kept"], true);

    host.api(
        "POST",
        "/v1/memory",
        Some(json!({
            "action": "delete",
            "memory_id": "slot:1",
        })),
    )
    .expect("forget LocalMobile memory");
    let forgotten = host.api("GET", "/v1/status", None).unwrap();
    assert_eq!(
        forgotten["mind"]["memories"]
            .as_array()
            .expect("memory projection")
            .len(),
        1
    );

    let updated = host
        .api(
            "PUT",
            "/v1/profile",
            Some(json!({
                "preferred_name": "Nolane User",
                "language": "en",
                "response_length": "balanced",
                "conversation_style": "direct",
                "initiative": "active",
                "memory_enabled": false,
                "personal_instruction": "Be concrete.",
            })),
        )
        .expect("update LocalMobile profile");
    assert_eq!(updated["preferred_name"], "Nolane User");
    assert_eq!(updated["language"], "en");
    assert_eq!(updated["initiative"], "active");
    assert_eq!(updated["memory_enabled"], false);

    drop(host);

    let mut reloaded = LocalMobileProductRuntime::load_for_court(
        &bundle,
        data.path(),
    )
    .expect("reload LocalMobile product host for court");

    let status = reloaded
        .api("GET", "/v1/status", None)
        .expect("reloaded status");
    assert_eq!(status["phase"], "off");
    assert_eq!(status["state_version"], 1);
    assert_eq!(status["interactions"], 18);
    assert_eq!(status["initiative"], "active");
    assert_eq!(status["memory_enabled"], false);
    let reloaded_memories = status["mind"]["memories"]
        .as_array()
        .expect("reloaded memories");
    assert_eq!(reloaded_memories.len(), 1);
    assert_eq!(
        reloaded_memories[0]["text"],
        "Người dùng đang xây Nolane v1."
    );
    assert_eq!(reloaded_memories[0]["kept"], true);

    let history = reloaded
        .api("GET", "/v1/history", None)
        .expect("reloaded history");
    assert_eq!(
        history["messages"].as_array().expect("message array").len(),
        2
    );

    let profile = reloaded
        .api("GET", "/v1/profile", None)
        .expect("reloaded profile");
    assert_eq!(profile["preferred_name"], "Nolane User");
    assert_eq!(profile["language"], "en");
    assert_eq!(profile["conversation_style"], "direct");
    assert_eq!(profile["personal_instruction"], "Be concrete.");

    let learning = reloaded.api(
        "GET",
        "/v1/learning/windows",
        None,
    );
    assert!(learning.is_err());

    println!("NOLANE_V056_LOCALMOBILE_PRODUCT_HOST_PASS");
}
