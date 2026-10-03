use nolane_mobile_runtime::product::LocalMobileProductRuntime;
use serde_json::json;
use std::{env, path::PathBuf};
use tempfile::tempdir;

fn main() {
    let mut args = env::args().skip(1);
    let bundle = PathBuf::from(
        args.next().expect("authorized bundle argument required"),
    );
    assert!(args.next().is_none(), "unexpected extra arguments");

    let data = tempdir().expect("create LocalMobile data dir");
    let mut host = LocalMobileProductRuntime::load(
        &bundle,
        data.path(),
    )
    .expect("strict LocalMobile loader must accept staged L36 bundle");

    let status = host
        .api("GET", "/v1/status", None)
        .expect("authorized status");
    assert_eq!(status["phase"], "off");
    assert_eq!(status["device"], "android-local-rust");
    assert!(
        status["identity_id"]
            .as_str()
            .expect("identity")
            .starts_with("nolane-mobile-")
    );

    host.api(
        "POST",
        "/v1/power",
        Some(json!({"enabled": true})),
    )
    .expect("authorized power on");

    let reply = host
        .api(
            "POST",
            "/v1/chat",
            Some(json!({"text": "Tiếp tục Nolane nhé"})),
        )
        .expect("authorized local chat");
    assert_eq!(reply["state_version"], 1);

    drop(host);

    let mut reloaded = LocalMobileProductRuntime::load(
        &bundle,
        data.path(),
    )
    .expect("authorized bundle must survive restart");
    let status = reloaded
        .api("GET", "/v1/status", None)
        .expect("reloaded authorized status");
    assert_eq!(status["state_version"], 1);
    assert_eq!(status["interactions"], 18);

    println!("NOLANE_V057_AUTHORIZED_LOCALMOBILE_RELEASE_PASS");
}
