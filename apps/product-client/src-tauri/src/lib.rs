use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Child,
    sync::Mutex,
    time::Duration,
};
use tauri::{Manager, RunEvent, State};
use url::Url;

#[cfg(target_os = "windows")]
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream},
    os::windows::process::CommandExt,
    process::{Command, Stdio},
    time::Instant,
};

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Debug)]
enum RuntimeTarget {
    Local { endpoint: String, token: String },
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
}

struct ProductState {
    manager: Mutex<RuntimeManager>,
    client: reqwest::Client,
    data_dir: PathBuf,
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

    if !runtime_exe.is_file() {
        return Err(format!(
            "Release runtime is missing: {}",
            runtime_exe.display()
        ));
    }
    if !model_checkpoint.is_file() {
        return Err(format!(
            "Release model is missing: {}",
            model_checkpoint.display()
        ));
    }
    if !tokenizer_dir.is_dir() {
        return Err(format!(
            "Release tokenizer is missing: {}",
            tokenizer_dir.display()
        ));
    }
    if !ceremony_path.is_file() {
        return Err(format!(
            "Release promotion ceremony is missing: {}",
            ceremony_path.display()
        ));
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
        .arg(data_dir)
        .arg("--model-bundle")
        .arg(&model_dir)
        .arg("--tokenizer")
        .arg(&tokenizer_dir)
        .arg("--ceremony")
        .arg(&ceremony_path)
        .arg("--device")
        .arg("auto")
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

#[cfg(not(target_os = "windows"))]
fn initial_non_windows_target(data_dir: &Path) -> RuntimeTarget {
    RuntimeTarget::Unconfigured {
        endpoint_hint: load_endpoint_hint(data_dir),
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

    #[cfg(not(target_os = "windows"))]
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
        manager.target = RuntimeTarget::Unconfigured {
            endpoint_hint: None,
        };
        Ok(manager.target.view())
    }
}

fn stop_child(state: &ProductState) {
    if let Ok(mut manager) = state.manager.lock() {
        if let Some(mut child) = manager.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
            fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
            let manager = initial_manager(app, &data_dir);
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(180))
                .build()
                .map_err(|e| e.to_string())?;
            app.manage(ProductState {
                manager: Mutex::new(manager),
                client,
                data_dir,
            });
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
