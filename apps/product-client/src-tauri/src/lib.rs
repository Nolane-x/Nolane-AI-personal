use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Child,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
        Mutex,
    },
    time::Duration,
};
use tauri::{Manager, RunEvent, State};
use url::Url;

#[cfg(target_os = "android")]
use nolane_mobile_runtime::product::{
    lifecycle_tick_seconds,
    LocalMobileProductRuntime,
};
#[cfg(target_os = "android")]
use tauri_plugin_fs::FsExt;

#[cfg(target_os = "windows")]
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream},
    os::windows::process::CommandExt,
    process::{Command, Stdio},
    time::Instant,
};

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(target_os = "android")]
const V058_COURT_TRANSACTION_ID: &str =
    "synthetic-mobile-release-court";
#[cfg(target_os = "android")]
const V058_COURT_CHECKPOINT_SHA256: &str =
    "7777777777777777777777777777777777777777777777777777777777777777";
#[cfg(target_os = "android")]
const V058_COURT_RECEIPT_SCHEMA: &str =
    "NOLANE-V058-ANDROID-EMULATOR-COURT-V1";

#[derive(Clone, Debug)]
enum RuntimeTarget {
    Local { endpoint: String, token: String },
    LocalMobile,
    Remote { endpoint: String, token: String },
    Unconfigured { endpoint_hint: Option<String> },
    Error { message: String },
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeTargetView {
    mode: String,
    endpoint: Option<String>,
    error: Option<String>,
}

impl RuntimeTarget {
    fn view(&self) -> RuntimeTargetView {
        match self {
            Self::Local { endpoint, .. } => RuntimeTargetView {
                mode: "local".into(),
                endpoint: Some(endpoint.clone()),
                error: None,
            },
            Self::LocalMobile => RuntimeTargetView {
                mode: "local-mobile".into(),
                endpoint: None,
                error: None,
            },
            Self::Remote { endpoint, .. } => RuntimeTargetView {
                mode: "remote".into(),
                endpoint: Some(endpoint.clone()),
                error: None,
            },
            Self::Unconfigured { endpoint_hint } => RuntimeTargetView {
                mode: "unconfigured".into(),
                endpoint: endpoint_hint.clone(),
                error: None,
            },
            Self::Error { message } => RuntimeTargetView {
                mode: "error".into(),
                endpoint: None,
                error: Some(message.clone()),
            },
        }
    }

    fn request_parts(&self) -> Result<(String, Option<String>), String> {
        match self {
            Self::Local { endpoint, token } => {
                Ok((endpoint.clone(), Some(token.clone())))
            },
            Self::Remote { endpoint, token } => {
                Ok((endpoint.clone(), Some(token.clone())))
            }
            Self::LocalMobile => {
                Err("LocalMobile must use the native Tauri route".into())
            }
            Self::Unconfigured { .. } => {
                Err("Android runtime is not paired yet".into())
            }
            Self::Error { message } => Err(message.clone()),
        }
    }
}

struct RuntimeManager {
    target: RuntimeTarget,
    child: Option<Child>,
    #[cfg(target_os = "android")]
    mobile: Option<LocalMobileProductRuntime>,
    #[cfg(target_os = "android")]
    v058_court_enabled: bool,
}

struct ProductState {
    manager: Mutex<RuntimeManager>,
    client: reqwest::Client,
    data_dir: PathBuf,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EndpointConfig {
    endpoint: String,
}

fn endpoint_config_path(data_dir: &Path) -> PathBuf {
    data_dir.join("remote-endpoint.json")
}

fn load_endpoint_hint(data_dir: &Path) -> Option<String> {
    let path = endpoint_config_path(data_dir);
    let text = fs::read_to_string(path).ok()?;
    let parsed: EndpointConfig = serde_json::from_str(&text).ok()?;
    Some(parsed.endpoint)
}

fn save_endpoint_hint(data_dir: &Path, endpoint: &str) -> Result<(), String> {
    fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let path = endpoint_config_path(data_dir);
    let temp = path.with_extension("json.tmp");
    let payload = serde_json::to_vec(&EndpointConfig {
        endpoint: endpoint.to_owned(),
    })
    .map_err(|e| e.to_string())?;
    fs::write(&temp, payload).map_err(|e| e.to_string())?;
    fs::rename(&temp, &path).map_err(|e| e.to_string())
}

fn validate_remote_endpoint(value: &str) -> Result<String, String> {
    let url = Url::parse(value.trim()).map_err(|_| "Invalid runtime endpoint")?;
    let scheme = url.scheme();
    if !matches!(scheme, "https" | "http") {
        return Err("Runtime endpoint must use https".into());
    }
    let host = url
        .host_str()
        .ok_or_else(|| "Runtime endpoint is missing a host".to_string())?;
    let loopback = matches!(host, "localhost" | "127.0.0.1" | "::1");
    if scheme != "https" && !loopback {
        return Err(
            "Remote runtime must use HTTPS so the pairing token is not exposed"
                .into(),
        );
    }
    if url.username() != "" || url.password().is_some() {
        return Err("Runtime endpoint must not contain embedded credentials".into());
    }
    let normalized = value.trim().trim_end_matches('/').to_owned();
    Ok(normalized)
}

fn validate_api_path(path: &str) -> Result<&str, String> {
    if !path.starts_with("/v1/") || path.contains("..") || path.contains('\0') {
        return Err("Unsupported product API path".into());
    }
    Ok(path)
}

#[cfg(target_os = "windows")]
fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(180)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(120));
    }
    false
}

#[cfg(target_os = "windows")]
fn spawn_windows_runtime(
    app: &tauri::App,
    data_dir: &Path,
) -> Result<(RuntimeTarget, Child), String> {
    let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;
    let runtime_exe = resource_dir
        .join("resources")
        .join("runtime")
        .join("nolane-product-runtime.exe");
    let model_dir = resource_dir.join("resources").join("model");
    let model_checkpoint = model_dir.join("factorized-nolane.pt");
    let tokenizer_dir = resource_dir.join("resources").join("tokenizer");
    let ceremony_path = model_dir.join("promotion-ceremony.json");
    let software_model = model_dir.join("Qwen3-0.6B-Q8_0.gguf");
    let software_manifest = model_dir.join("software-release.json");
    let llama_server = resource_dir
        .join("resources")
        .join("runtime")
        .join("llama")
        .join("llama-server.exe");

    if !runtime_exe.is_file() {
        return Err(format!(
            "Release runtime is missing: {}",
            runtime_exe.display()
        ));
    }

    let certified_ready = model_checkpoint.is_file()
        && tokenizer_dir.is_dir()
        && ceremony_path.is_file();
    let software_ready = software_model.is_file()
        && software_manifest.is_file()
        && llama_server.is_file();

    if !certified_ready && !software_ready {
        return Err(
            "No complete Windows inference channel is bundled. Expected either "
                .to_string()
                + "factorized model + tokenizer + COMPLETE ceremony, or "
                + "pinned GGUF model + llama.cpp + software manifest.",
        );
    }

    let port = portpicker::pick_unused_port()
        .ok_or_else(|| "Could not allocate a local Nolane runtime port".to_string())?;
    let mut token_bytes = [0u8; 32];
    OsRng.fill_bytes(&mut token_bytes);
    let auth_token = token_bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let mut command = Command::new(&runtime_exe);
    command
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--data-dir")
        .arg(data_dir);

    if certified_ready {
        command
            .arg("--model-bundle")
            .arg(&model_dir)
            .arg("--tokenizer")
            .arg(&tokenizer_dir)
            .arg("--ceremony")
            .arg(&ceremony_path)
            .arg("--device")
            .arg("auto");
    } else {
        command
            .arg("--software-model")
            .arg(&software_model)
            .arg("--llama-server")
            .arg(&llama_server)
            .arg("--software-manifest")
            .arg(&software_manifest)
            .arg("--device")
            .arg("cpu");
    }

    command
        .arg("--auth-token")
        .arg(&auth_token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);

    let mut child = command.spawn().map_err(|e| {
        format!("Could not start the bundled Nolane runtime: {e}")
    })?;

    if !wait_for_port(port, Duration::from_secs(12)) {
        let _ = child.kill();
        let _ = child.wait();
        return Err("Bundled Nolane runtime did not become ready".into());
    }

    Ok((
        RuntimeTarget::Local {
            endpoint: format!("http://127.0.0.1:{port}"),
            token: auth_token,
        },
        child,
    ))
}

#[cfg(all(not(target_os = "windows"), not(target_os = "android")))]
fn initial_non_windows_target(data_dir: &Path) -> RuntimeTarget {
    RuntimeTarget::Unconfigured {
        endpoint_hint: load_endpoint_hint(data_dir),
    }
}

#[cfg(target_os = "android")]
fn android_asset_bytes(
    app: &tauri::App,
    relative: &str,
) -> Result<Vec<u8>, String> {
    let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;
    let path = resource_dir.join(relative);
    app.fs().read(path).map_err(|error| {
        format!("Could not read packaged Android asset {relative}: {error}")
    })
}

#[cfg(target_os = "android")]
fn safe_bundle_filename(value: &str, label: &str) -> Result<String, String> {
    if value.is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value == "."
        || value == ".."
    {
        return Err(format!("Invalid LocalMobile {label} filename"));
    }
    Ok(value.to_string())
}

#[cfg(target_os = "android")]
fn write_android_asset(
    app: &tauri::App,
    relative: &str,
    destination: &Path,
) -> Result<(), String> {
    let bytes = android_asset_bytes(app, relative)?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(destination, bytes).map_err(|e| e.to_string())
}

#[cfg(target_os = "android")]
fn materialize_android_mobile_bundle(
    app: &tauri::App,
    data_dir: &Path,
) -> Result<(PathBuf, bool), String> {
    const ROOT: &str = "resources/mobile";

    let manifest_bytes = android_asset_bytes(
        app,
        &format!("{ROOT}/localmobile-manifest.json"),
    )?;
    let manifest: Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let checkpoint = manifest
        .get("source_checkpoint_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "Packaged LocalMobile manifest lacks checkpoint".to_string())?;
    let ceremony_sha = manifest
        .get("promotion_ceremony_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "Packaged LocalMobile manifest lacks ceremony digest".to_string())?;
    if checkpoint.len() != 64
        || ceremony_sha.len() != 64
        || !checkpoint.bytes().all(|b| b.is_ascii_hexdigit())
        || !ceremony_sha.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("Packaged LocalMobile bundle digest is malformed".into());
    }

    let ceremony_bytes = android_asset_bytes(
        app,
        &format!("{ROOT}/promotion-ceremony.json"),
    )?;
    let ceremony: Value =
        serde_json::from_slice(&ceremony_bytes).map_err(|e| e.to_string())?;
    let court_enabled = ceremony
        .get("transaction_id")
        .and_then(Value::as_str)
        == Some(V058_COURT_TRANSACTION_ID)
        && ceremony
            .get("candidate_checkpoint_sha256")
            .and_then(Value::as_str)
            == Some(V058_COURT_CHECKPOINT_SHA256);

    let key = format!(
        "{}-{}",
        &checkpoint[..16],
        &ceremony_sha[..16],
    );
    let cache_root = data_dir.join("packaged-localmobile");
    let destination = cache_root.join(&key);
    if destination.join("localmobile-manifest.json").is_file() {
        return Ok((destination, court_enabled));
    }

    fs::create_dir_all(&cache_root).map_err(|e| e.to_string())?;
    let staging = cache_root.join(format!(".{key}.staging"));
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(staging.join("package")).map_err(|e| e.to_string())?;

    fs::write(staging.join("localmobile-manifest.json"), manifest_bytes)
        .map_err(|e| e.to_string())?;
    fs::write(staging.join("promotion-ceremony.json"), ceremony_bytes)
        .map_err(|e| e.to_string())?;

    for name in [
        "bootstrap-state.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "prompt-contract.json",
    ] {
        write_android_asset(
            app,
            &format!("{ROOT}/{name}"),
            &staging.join(name),
        )?;
    }

    let package_manifest_bytes = android_asset_bytes(
        app,
        &format!("{ROOT}/package/manifest.json"),
    )?;
    let package_manifest: Value = serde_json::from_slice(
        &package_manifest_bytes,
    )
    .map_err(|e| e.to_string())?;
    let contract_filename = safe_bundle_filename(
        package_manifest
            .get("contract_filename")
            .and_then(Value::as_str)
            .ok_or_else(|| "Mobile package lacks contract filename".to_string())?,
        "contract",
    )?;
    let weights_filename = safe_bundle_filename(
        package_manifest
            .get("weights_filename")
            .and_then(Value::as_str)
            .ok_or_else(|| "Mobile package lacks weights filename".to_string())?,
        "weights",
    )?;
    fs::write(
        staging.join("package").join("manifest.json"),
        package_manifest_bytes,
    )
    .map_err(|e| e.to_string())?;
    for name in [&contract_filename, &weights_filename] {
        write_android_asset(
            app,
            &format!("{ROOT}/package/{name}"),
            &staging.join("package").join(name),
        )?;
    }

    if destination.exists() {
        fs::remove_dir_all(&destination).map_err(|e| e.to_string())?;
    }
    fs::rename(&staging, &destination).map_err(|e| e.to_string())?;
    Ok((destination, court_enabled))
}

#[cfg(target_os = "android")]
fn v058_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("v0.58 court missing numeric field: {key}"))
}

#[cfg(target_os = "android")]
fn v058_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("v0.58 court missing text field: {key}"))
}

#[cfg(target_os = "android")]
fn v058_history_len(value: &Value) -> Result<u64, String> {
    value
        .get("messages")
        .and_then(Value::as_array)
        .map(|rows| rows.len() as u64)
        .ok_or_else(|| "v0.58 court history payload is invalid".to_string())
}

#[cfg(target_os = "android")]
fn write_v058_court_receipt(
    path: &Path,
    payload: &Value,
) -> Result<(), String> {
    let temp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec(payload).map_err(|e| e.to_string())?;
    fs::write(&temp, bytes).map_err(|e| e.to_string())?;
    fs::rename(&temp, path).map_err(|e| e.to_string())
}

#[cfg(target_os = "android")]
fn run_v058_android_emulator_court(
    manager: &mut RuntimeManager,
    data_dir: &Path,
) -> Result<(), String> {
    if !manager.v058_court_enabled {
        return Ok(());
    }
    if !matches!(manager.target, RuntimeTarget::LocalMobile) {
        return Err(
            "v0.58 synthetic emulator court did not boot LocalMobile".into(),
        );
    }
    let mobile = manager.mobile.as_mut().ok_or_else(|| {
        "v0.58 synthetic emulator court has no native runtime".to_string()
    })?;

    let receipt_path = data_dir.join("v058-android-emulator-court.json");
    let previous = if receipt_path.is_file() {
        Some(
            serde_json::from_slice::<Value>(
                &fs::read(&receipt_path).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    let before = mobile
        .api("GET", "/v1/status", None)
        .map_err(|e| e.to_string())?;
    let history_before = mobile
        .api("GET", "/v1/history", None)
        .map_err(|e| e.to_string())?;
    let identity = v058_string(&before, "identity_id")?;
    if !identity.starts_with("nolane-mobile-") {
        return Err("v0.58 court identity is not device-local".into());
    }
    if before.get("phase").and_then(Value::as_str) != Some("off") {
        return Err("v0.58 court expected power state off after process boot".into());
    }

    let before_version = v058_u64(&before, "state_version")?;
    let before_interactions = v058_u64(&before, "interactions")?;
    let before_history = v058_history_len(&history_before)?;

    let expected_stage = previous
        .as_ref()
        .and_then(|value| value.get("stage"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if let Some(previous) = previous.as_ref() {
        if previous.get("schema").and_then(Value::as_str)
            != Some(V058_COURT_RECEIPT_SCHEMA)
        {
            return Err("v0.58 court receipt schema mismatch".into());
        }
        let previous_identity = v058_string(previous, "identity_id")?;
        let previous_version = v058_u64(previous, "state_version")?;
        let previous_interactions = v058_u64(previous, "interactions")?;
        let previous_history = v058_u64(previous, "history_len")?;
        if previous_identity != identity {
            return Err("v0.58 identity changed across app restart".into());
        }
        if previous_version != before_version
            || previous_interactions != before_interactions
            || previous_history != before_history
        {
            return Err(
                "v0.58 persisted state/history drifted across app restart".into(),
            );
        }
    } else if before_version != 0 || before_history != 0 {
        return Err("v0.58 first boot was not a clean LocalMobile state".into());
    }

    if expected_stage >= 2 {
        log::info!(
            "NOLANE_V058_EMULATOR_FINAL_PASS identity={} state_version={} history={}",
            identity,
            before_version,
            before_history,
        );
        return Ok(());
    }

    mobile
        .api(
            "POST",
            "/v1/power",
            Some(json!({"enabled": true})),
        )
        .map_err(|e| e.to_string())?;
    let prompt = if expected_stage == 0 {
        "v058 emulator first boot local chat"
    } else {
        "v058 emulator restart local chat"
    };
    // Exercise the same native product prompt/kernel/sampler/state path as
    // /v1/chat, but keep this synthetic emulator proof bounded so CI measures
    // packaged correctness rather than spending minutes generating 96 tokens.
    let reply = mobile
        .synthetic_court_chat(prompt, 8)
        .map_err(|e| e.to_string())?;
    if reply.get("reply").and_then(Value::as_str).is_none() {
        return Err("v0.58 local chat did not return a reply field".into());
    }

    let after = mobile
        .api("GET", "/v1/status", None)
        .map_err(|e| e.to_string())?;
    let history_after = mobile
        .api("GET", "/v1/history", None)
        .map_err(|e| e.to_string())?;
    let after_identity = v058_string(&after, "identity_id")?;
    let after_version = v058_u64(&after, "state_version")?;
    let after_interactions = v058_u64(&after, "interactions")?;
    let after_history = v058_history_len(&history_after)?;

    if after_identity != identity
        || after_version != before_version.saturating_add(1)
        || after_interactions != before_interactions.saturating_add(1)
        || after_history != before_history.saturating_add(2)
    {
        return Err("v0.58 local chat state transition mismatch".into());
    }

    let stage = expected_stage + 1;
    let receipt = json!({
        "schema": V058_COURT_RECEIPT_SCHEMA,
        "stage": stage,
        "identity_id": identity,
        "state_version": after_version,
        "interactions": after_interactions,
        "history_len": after_history,
        "checkpoint_sha256": V058_COURT_CHECKPOINT_SHA256,
    });
    write_v058_court_receipt(&receipt_path, &receipt)?;

    if stage == 1 {
        log::info!(
            "NOLANE_V058_FIRST_BOOT_PASS identity={} state_version={} history={}",
            after_identity,
            after_version,
            after_history,
        );
    } else {
        log::info!(
            "NOLANE_V058_RESTART_PASS identity={} state_version={} history={}",
            after_identity,
            after_version,
            after_history,
        );
    }
    Ok(())
}

#[cfg(target_os = "android")]
fn load_android_local_mobile(
    app: &tauri::App,
    data_dir: &Path,
) -> Result<(LocalMobileProductRuntime, bool), String> {
    let (bundle_dir, court_enabled) =
        materialize_android_mobile_bundle(app, data_dir)?;
    let mobile_data = data_dir.join("local-mobile");

    let load = || {
        LocalMobileProductRuntime::load(&bundle_dir, &mobile_data).map_err(|error| {
            format!(
                "LocalMobile runtime unavailable from {}: {}",
                bundle_dir.display(),
                error
            )
        })
    };
    match load() {
        Ok(mobile) => Ok((mobile, court_enabled)),
        Err(first_error) => {
            // Cached APK assets are derived, not user state. If the cache was
            // interrupted or corrupted, rebuild it once from immutable assets.
            let _ = fs::remove_dir_all(&bundle_dir);
            let (rebuilt_dir, rebuilt_court_enabled) =
                materialize_android_mobile_bundle(app, data_dir)?;
            LocalMobileProductRuntime::load(&rebuilt_dir, &mobile_data)
                .map(|mobile| (mobile, rebuilt_court_enabled))
                .map_err(|second| {
                    format!("{first_error}; rebuild failed: {second}")
                })
        }
    }
}

fn initial_manager(
    app: &tauri::App,
    data_dir: &Path,
) -> RuntimeManager {
    #[cfg(target_os = "windows")]
    {
        return match spawn_windows_runtime(app, data_dir) {
            Ok((target, child)) => RuntimeManager {
                target,
                child: Some(child),
            },
            Err(message) => RuntimeManager {
                target: RuntimeTarget::Error { message },
                child: None,
            },
        };
    }

    #[cfg(target_os = "android")]
    {
        return match load_android_local_mobile(app, data_dir) {
            Ok((mobile, v058_court_enabled)) => RuntimeManager {
                target: RuntimeTarget::LocalMobile,
                child: None,
                mobile: Some(mobile),
                v058_court_enabled,
            },
            Err(message) => RuntimeManager {
                target: RuntimeTarget::Error { message },
                child: None,
                mobile: None,
                v058_court_enabled: false,
            },
        };
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "android")))]
    {
        let _ = app;
        RuntimeManager {
            target: initial_non_windows_target(data_dir),
            child: None,
        }
    }
}

#[tauri::command]
fn runtime_target(state: State<'_, ProductState>) -> Result<RuntimeTargetView, String> {
    let manager = state
        .manager
        .lock()
        .map_err(|_| "Runtime state lock poisoned".to_string())?;
    Ok(manager.target.view())
}

#[tauri::command]
async fn product_api(
    state: State<'_, ProductState>,
    method: String,
    path: String,
    body: Option<Value>,
) -> Result<Value, String> {
    let path = validate_api_path(&path)?;

    #[cfg(target_os = "android")]
    {
        let mut manager = state
            .manager
            .lock()
            .map_err(|_| "Runtime state lock poisoned".to_string())?;
        if matches!(manager.target, RuntimeTarget::LocalMobile) {
            let mobile = manager.mobile.as_mut().ok_or_else(|| {
                "LocalMobile target has no native runtime".to_string()
            })?;
            return mobile
                .api(&method, path, body.clone())
                .map_err(|error| error.to_string());
        }
    }

    let (endpoint, token) = {
        let manager = state
            .manager
            .lock()
            .map_err(|_| "Runtime state lock poisoned".to_string())?;
        manager.target.request_parts()?
    };

    let url = format!("{}{}", endpoint.trim_end_matches('/'), path);
    let method = reqwest::Method::from_bytes(method.as_bytes())
        .map_err(|_| "Unsupported HTTP method".to_string())?;
    if !matches!(
        method,
        reqwest::Method::GET | reqwest::Method::POST | reqwest::Method::PUT
    ) {
        return Err("Unsupported product API method".into());
    }

    let mut request = state
        .client
        .request(method, &url)
        .header("Accept", "application/json");
    if let Some(value) = token {
        request = request.header("X-Nolane-Token", value);
    }
    if let Some(payload) = body {
        request = request.json(&payload);
    }

    let response = request
        .send()
        .await
        .map_err(|e| format!("Nolane runtime unavailable: {e}"))?;
    let status = response.status();
    let payload: Value = response
        .json()
        .await
        .map_err(|e| format!("Invalid Nolane runtime response: {e}"))?;

    if !status.is_success() {
        let message = payload
            .get("message")
            .or_else(|| payload.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("Nolane runtime request failed");
        return Err(message.to_owned());
    }
    Ok(payload)
}

#[tauri::command]
fn configure_remote(
    state: State<'_, ProductState>,
    endpoint: String,
    token: String,
) -> Result<RuntimeTargetView, String> {
    #[cfg(target_os = "windows")]
    {
        let _ = (state, endpoint, token);
        return Err("Windows uses the bundled local Nolane runtime".into());
    }

    #[cfg(not(target_os = "windows"))]
    {
        let endpoint = validate_remote_endpoint(&endpoint)?;
        if token.trim().len() < 16 {
            return Err("Pairing token is too short".into());
        }
        save_endpoint_hint(&state.data_dir, &endpoint)?;
        let mut manager = state
            .manager
            .lock()
            .map_err(|_| "Runtime state lock poisoned".to_string())?;
        manager.target = RuntimeTarget::Remote {
            endpoint,
            token: token.trim().to_owned(),
        };
        Ok(manager.target.view())
    }
}

#[tauri::command]
fn clear_remote(state: State<'_, ProductState>) -> Result<RuntimeTargetView, String> {
    #[cfg(target_os = "windows")]
    {
        let _ = state;
        return Err("Windows uses the bundled local Nolane runtime".into());
    }

    #[cfg(not(target_os = "windows"))]
    {
        let path = endpoint_config_path(&state.data_dir);
        if path.exists() {
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
        let mut manager = state
            .manager
            .lock()
            .map_err(|_| "Runtime state lock poisoned".to_string())?;
        #[cfg(target_os = "android")]
        {
            manager.target = if manager.mobile.is_some() {
                RuntimeTarget::LocalMobile
            } else {
                RuntimeTarget::Error {
                    message: "LocalMobile runtime is unavailable".into(),
                }
            };
        }
        #[cfg(not(target_os = "android"))]
        {
            manager.target = RuntimeTarget::Unconfigured {
                endpoint_hint: None,
            };
        }
        Ok(manager.target.view())
    }
}

fn stop_child(state: &ProductState) {
    state.shutdown.store(true, Ordering::Relaxed);
    if let Ok(mut manager) = state.manager.lock() {
        if let Some(mut child) = manager.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(target_os = "android")]
fn start_android_lifecycle_ticker(
    app_handle: tauri::AppHandle,
    shutdown: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            let delay = {
                let state = app_handle.state::<ProductState>();
                let mut manager = match state.manager.lock() {
                    Ok(value) => value,
                    Err(_) => return,
                };
                if !matches!(manager.target, RuntimeTarget::LocalMobile) {
                    Duration::from_secs(5)
                } else if let Some(mobile) = manager.mobile.as_mut() {
                    match mobile.api("GET", "/v1/profile", None) {
                        Ok(profile) => Duration::from_secs(
                            lifecycle_tick_seconds(
                                profile
                                    .get("initiative")
                                    .and_then(Value::as_str)
                                    .unwrap_or("gentle"),
                            ),
                        ),
                        Err(_) => Duration::from_secs(
                            lifecycle_tick_seconds("gentle"),
                        ),
                    }
                } else {
                    Duration::from_secs(5)
                }
            };

            let mut elapsed = Duration::ZERO;
            while elapsed < delay && !shutdown.load(Ordering::Relaxed) {
                let slice = (delay - elapsed).min(Duration::from_secs(1));
                std::thread::sleep(slice);
                elapsed += slice;
            }
            if shutdown.load(Ordering::Relaxed) {
                return;
            }

            let state = app_handle.state::<ProductState>();
            let mut manager = match state.manager.lock() {
                Ok(value) => value,
                Err(_) => return,
            };
            if !matches!(manager.target, RuntimeTarget::LocalMobile) {
                continue;
            }
            let Some(mobile) = manager.mobile.as_mut() else {
                continue;
            };
            if let Err(error) = mobile.api("POST", "/v1/tick", None) {
                log::warn!("LocalMobile lifecycle tick failed: {error}");
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "android")]
    android_logger::init_once(
        android_logger::Config::default()
            .with_tag("Nolane")
            .with_max_level(log::LevelFilter::Info),
    );

    let builder = tauri::Builder::default();
    #[cfg(target_os = "android")]
    let builder = builder.plugin(tauri_plugin_fs::init());

    let app = builder
        .setup(|app| {
            let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
            fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
            let mut manager = initial_manager(app, &data_dir);
            #[cfg(target_os = "android")]
            if let Err(error) = run_v058_android_emulator_court(
                &mut manager,
                &data_dir,
            ) {
                log::error!("NOLANE_V058_EMULATOR_COURT_FAIL {error}");
                return Err(error.into());
            }
            #[cfg(target_os = "android")]
            let start_mobile_ticker = !manager.v058_court_enabled;

            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(180))
                .build()
                .map_err(|e| e.to_string())?;
            let shutdown = Arc::new(AtomicBool::new(false));
            let ticker_shutdown = Arc::clone(&shutdown);
            let app_handle = app.handle().clone();
            app.manage(ProductState {
                manager: Mutex::new(manager),
                client,
                data_dir,
                shutdown,
            });

            #[cfg(target_os = "android")]
            if start_mobile_ticker {
                start_android_lifecycle_ticker(
                    app_handle,
                    ticker_shutdown,
                );
            }
            #[cfg(not(target_os = "android"))]
            {
                let _ = (app_handle, ticker_shutdown);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            runtime_target,
            product_api,
            configure_remote,
            clear_remote
        ])
        .build(tauri::generate_context!())
        .expect("failed to build Nolane product client");

    app.run(|app_handle, event| {
        if let RunEvent::Exit { .. } = event {
            let state = app_handle.state::<ProductState>();
            stop_child(&state);
        }
    });
}
