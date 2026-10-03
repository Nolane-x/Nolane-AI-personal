use crate::{
    read_persistent_mobile_state,
    write_persistent_mobile_state,
    MobileRuntime,
    PersistentMobileState,
    RuntimeError,
    PRODUCT_SAMPLING_TEMPERATURE,
    PRODUCT_SAMPLING_TOP_P,
};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const LOCAL_MOBILE_BUNDLE_SCHEMA: &str =
    "NOLANE-V057-AUTHORIZED-LOCALMOBILE-BUNDLE-V1";
pub const LOCAL_MOBILE_RELEASE_AUTHORITY: &str =
    "L36_COMPLETE_PROMOTION_CEREMONY";
pub const LOCAL_MOBILE_COURT_AUTHORITY: &str =
    "SYNTHETIC_COURT_ONLY_NO_RELEASE_AUTHORITY";
pub const LOCAL_MOBILE_META_SCHEMA: &str =
    "NOLANE-V056-LOCALMOBILE-META-V1";
const MAX_HISTORY_MESSAGES: usize = 400;
const MAX_PROFILE_NAME_CHARS: usize = 80;
const MAX_PERSONAL_INSTRUCTION_CHARS: usize = 1200;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LocalMobileBundleManifest {
    pub schema: String,
    pub authority: String,
    pub source_checkpoint_sha256: String,
    pub prompt_contract_file_sha256: String,
    pub mobile_package_manifest_sha256: String,
    pub tokenizer_json_sha256: String,
    pub tokenizer_config_json_sha256: String,
    pub bootstrap_state_sha256: String,
    pub promotion_ceremony_sha256: String,
    pub promotion_authorization_sha256: String,
    pub promotion_ceremony_file_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct PromotionCeremonyView {
    schema: String,
    authority: String,
    status: String,
    candidate_checkpoint_sha256: String,
    authorization_sha256: String,
    ceremony_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LocalMobileMeta {
    schema: String,
    state_version: u64,
    memory_enabled: bool,
    initiative: String,
}

impl Default for LocalMobileMeta {
    fn default() -> Self {
        Self {
            schema: LOCAL_MOBILE_META_SCHEMA.to_string(),
            state_version: 0,
            memory_enabled: true,
            initiative: "gentle".to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LocalMobileMessage {
    event_id: String,
    at: String,
    role: String,
    text: String,
}

pub struct LocalMobileProductRuntime {
    runtime: MobileRuntime,
    data_dir: PathBuf,
    state_path: PathBuf,
    meta_path: PathBuf,
    history_path: PathBuf,
    meta: LocalMobileMeta,
    history: Vec<LocalMobileMessage>,
    powered: bool,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), RuntimeError> {
    let parent = path.parent().ok_or_else(|| {
        RuntimeError::Invalid("LocalMobile path has no parent".into())
    })?;
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| RuntimeError::Invalid(
            "LocalMobile path has invalid file name".into()
        ))?;
    let mut nonce = [0u8; 8];
    OsRng.fill_bytes(&mut nonce);
    let suffix = nonce
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let temporary = parent.join(format!(".{file_name}.{suffix}.tmp"));
    let result = (|| -> Result<(), RuntimeError> {
        let mut handle = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        handle.write_all(bytes)?;
        handle.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), RuntimeError> {
    let bytes = serde_json::to_vec(value)?;
    atomic_write(path, &bytes)
}

fn now_marker() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    millis.to_string()
}

fn fresh_identity() -> String {
    let mut bytes = [0u8; 16];
    OsRng.fill_bytes(&mut bytes);
    let suffix = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("nolane-mobile-{suffix}")
}

fn verify_bundle_file(
    path: &Path,
    expected_sha256: &str,
    label: &str,
) -> Result<(), RuntimeError> {
    if !is_lower_hex_sha256(expected_sha256) {
        return Err(RuntimeError::Invalid(format!(
            "LocalMobile {label} digest is malformed"
        )));
    }
    let actual = sha256_hex(&fs::read(path)?);
    if actual != expected_sha256 {
        return Err(RuntimeError::Invalid(format!(
            "LocalMobile {label} SHA-256 mismatch"
        )));
    }
    Ok(())
}

fn load_manifest(
    bundle_dir: &Path,
    allow_court_authority: bool,
) -> Result<LocalMobileBundleManifest, RuntimeError> {
    let bytes = fs::read(bundle_dir.join("localmobile-manifest.json"))?;
    let manifest: LocalMobileBundleManifest = serde_json::from_slice(&bytes)?;
    if manifest.schema != LOCAL_MOBILE_BUNDLE_SCHEMA {
        return Err(RuntimeError::Invalid(
            "LocalMobile bundle schema mismatch".into(),
        ));
    }
    let authorized = manifest.authority == LOCAL_MOBILE_RELEASE_AUTHORITY;
    let court = manifest.authority == LOCAL_MOBILE_COURT_AUTHORITY;
    if !authorized && !(allow_court_authority && court) {
        return Err(RuntimeError::Invalid(
            "LocalMobile bundle lacks L36 release authority".into(),
        ));
    }
    for (name, value) in [
        ("source checkpoint", manifest.source_checkpoint_sha256.as_str()),
        ("prompt contract", manifest.prompt_contract_file_sha256.as_str()),
        ("mobile package manifest", manifest.mobile_package_manifest_sha256.as_str()),
        ("tokenizer.json", manifest.tokenizer_json_sha256.as_str()),
        ("tokenizer_config.json", manifest.tokenizer_config_json_sha256.as_str()),
        ("bootstrap state", manifest.bootstrap_state_sha256.as_str()),
        ("promotion ceremony", manifest.promotion_ceremony_sha256.as_str()),
        ("promotion authorization", manifest.promotion_authorization_sha256.as_str()),
        ("promotion ceremony file", manifest.promotion_ceremony_file_sha256.as_str()),
    ] {
        if !is_lower_hex_sha256(value) {
            return Err(RuntimeError::Invalid(format!(
                "LocalMobile {name} digest is malformed"
            )));
        }
    }

    verify_bundle_file(
        &bundle_dir.join("package").join("manifest.json"),
        &manifest.mobile_package_manifest_sha256,
        "mobile package manifest",
    )?;
    verify_bundle_file(
        &bundle_dir.join("tokenizer.json"),
        &manifest.tokenizer_json_sha256,
        "tokenizer.json",
    )?;
    verify_bundle_file(
        &bundle_dir.join("tokenizer_config.json"),
        &manifest.tokenizer_config_json_sha256,
        "tokenizer_config.json",
    )?;
    verify_bundle_file(
        &bundle_dir.join("bootstrap-state.json"),
        &manifest.bootstrap_state_sha256,
        "bootstrap state",
    )?;
    verify_bundle_file(
        &bundle_dir.join("prompt-contract.json"),
        &manifest.prompt_contract_file_sha256,
        "prompt contract",
    )?;

    if authorized {
        let ceremony_path = bundle_dir.join("promotion-ceremony.json");
        verify_bundle_file(
            &ceremony_path,
            &manifest.promotion_ceremony_file_sha256,
            "promotion ceremony file",
        )?;
        let ceremony: PromotionCeremonyView =
            serde_json::from_slice(&fs::read(&ceremony_path)?)?;
        if ceremony.schema != "NOLANE-L36-PROMOTION-CEREMONY-V1"
            || ceremony.authority != "FINAL_PROMOTION_CEREMONY_EVIDENCE"
            || ceremony.status != "COMPLETE"
        {
            return Err(RuntimeError::Invalid(
                "LocalMobile promotion ceremony is not COMPLETE L36 evidence".into(),
            ));
        }
        if ceremony.candidate_checkpoint_sha256
            != manifest.source_checkpoint_sha256
        {
            return Err(RuntimeError::Invalid(
                "LocalMobile ceremony checkpoint mismatch".into(),
            ));
        }
        if ceremony.authorization_sha256
            != manifest.promotion_authorization_sha256
        {
            return Err(RuntimeError::Invalid(
                "LocalMobile ceremony authorization mismatch".into(),
            ));
        }
        if ceremony.ceremony_sha256
            != manifest.promotion_ceremony_sha256
        {
            return Err(RuntimeError::Invalid(
                "LocalMobile ceremony digest mismatch".into(),
            ));
        }
    }

    let package_manifest: Value = serde_json::from_slice(
        &fs::read(bundle_dir.join("package").join("manifest.json"))?
    )?;
    if package_manifest
        .get("source_checkpoint_sha256")
        .and_then(Value::as_str)
        != Some(manifest.source_checkpoint_sha256.as_str())
    {
        return Err(RuntimeError::Invalid(
            "LocalMobile package checkpoint mismatch".into(),
        ));
    }
    Ok(manifest)
}

fn load_meta(path: &Path) -> Result<LocalMobileMeta, RuntimeError> {
    if !path.exists() {
        return Ok(LocalMobileMeta::default());
    }
    let meta: LocalMobileMeta = serde_json::from_slice(&fs::read(path)?)?;
    if meta.schema != LOCAL_MOBILE_META_SCHEMA {
        return Err(RuntimeError::Invalid(
            "LocalMobile metadata schema mismatch".into(),
        ));
    }
    if !matches!(meta.initiative.as_str(), "off" | "gentle" | "active") {
        return Err(RuntimeError::Invalid(
            "LocalMobile initiative value is invalid".into(),
        ));
    }
    Ok(meta)
}

fn load_history(path: &Path) -> Result<Vec<LocalMobileMessage>, RuntimeError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let history: Vec<LocalMobileMessage> =
        serde_json::from_slice(&fs::read(path)?)?;
    if history.len() > MAX_HISTORY_MESSAGES {
        return Err(RuntimeError::Invalid(
            "LocalMobile history exceeds bounded message limit".into(),
        ));
    }
    if history.iter().any(|message| {
        !matches!(message.role.as_str(), "user" | "assistant")
            || message.text.len() > 65_536
    }) {
        return Err(RuntimeError::Invalid(
            "LocalMobile history contains invalid message".into(),
        ));
    }
    Ok(history)
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.trim().chars().take(limit).collect()
}

impl LocalMobileProductRuntime {
    pub fn load(
        bundle_dir: impl AsRef<Path>,
        data_dir: impl AsRef<Path>,
    ) -> Result<Self, RuntimeError> {
        Self::load_inner(bundle_dir, data_dir, false)
    }

    pub fn load_for_court(
        bundle_dir: impl AsRef<Path>,
        data_dir: impl AsRef<Path>,
    ) -> Result<Self, RuntimeError> {
        Self::load_inner(bundle_dir, data_dir, true)
    }

    fn load_inner(
        bundle_dir: impl AsRef<Path>,
        data_dir: impl AsRef<Path>,
        allow_court_authority: bool,
    ) -> Result<Self, RuntimeError> {
        let bundle_dir = bundle_dir.as_ref();
        let data_dir = data_dir.as_ref().to_path_buf();
        fs::create_dir_all(&data_dir)?;

        let manifest = load_manifest(
            bundle_dir,
            allow_court_authority,
        )?;
        let state_path = data_dir.join("persistent-state.json");
        let meta_path = data_dir.join("local-mobile-meta.json");
        let history_path = data_dir.join("local-mobile-history.json");

        if !state_path.exists() {
            let mut bootstrap = read_persistent_mobile_state(
                bundle_dir.join("bootstrap-state.json"),
                Some(&manifest.source_checkpoint_sha256),
                None,
            )?;
            bootstrap.state.identity_id = fresh_identity();
            write_persistent_mobile_state(
                &state_path,
                &bootstrap,
                Some(&manifest.source_checkpoint_sha256),
                None,
            )?;
        }

        let runtime =
            MobileRuntime::load_with_prompt_contract_and_persistent_state(
                bundle_dir.join("package"),
                bundle_dir.join("tokenizer.json"),
                bundle_dir.join("tokenizer_config.json"),
                bundle_dir.join("prompt-contract.json"),
                &manifest.prompt_contract_file_sha256,
                Some(&manifest.source_checkpoint_sha256),
                &state_path,
            )?;

        let meta = load_meta(&meta_path)?;
        let history = load_history(&history_path)?;

        Ok(Self {
            runtime,
            data_dir,
            state_path,
            meta_path,
            history_path,
            meta,
            history,
            powered: false,
        })
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn api(
        &mut self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, RuntimeError> {
        let method = method.to_ascii_uppercase();
        let route = path.split('?').next().unwrap_or(path);
        match (method.as_str(), route) {
            ("GET", "/v1/status") => Ok(self.status()),
            ("GET", "/v1/readiness") => Ok(json!({
                "status": "PASS",
                "critical_failures": 0,
                "advisory_failures": 0,
                "target": "LocalMobile",
            })),
            ("GET", "/v1/history") => Ok(json!({
                "messages": &self.history,
            })),
            ("GET", "/v1/profile") => self.profile_json(),
            ("PUT", "/v1/profile") => {
                let patch = body.ok_or_else(|| RuntimeError::Invalid(
                    "profile update body is required".into(),
                ))?;
                self.update_profile(patch)
            }
            ("POST", "/v1/power") => {
                let enabled = body
                    .as_ref()
                    .and_then(|value| value.get("enabled"))
                    .and_then(Value::as_bool)
                    .ok_or_else(|| RuntimeError::Invalid(
                        "enabled must be boolean".into(),
                    ))?;
                self.powered = enabled;
                Ok(self.status())
            }
            ("POST", "/v1/chat") => {
                let text = body
                    .as_ref()
                    .and_then(|value| value.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.send_message(text)
            }
            (_, route) if route.starts_with("/v1/learning/") => {
                Err(RuntimeError::Invalid(
                    "Learning review is not available on LocalMobile".into(),
                ))
            }
            _ => Err(RuntimeError::Invalid(format!(
                "Unsupported LocalMobile product API route: {method} {path}"
            ))),
        }
    }

    fn status(&self) -> Value {
        let state = self.runtime.persistent_state();
        let identity_id = state
            .map(|value| value.state.identity_id.as_str())
            .unwrap_or("");
        let interactions = state
            .map(|value| value.state.relationship.interaction_count)
            .unwrap_or(0);
        let open_threads = state
            .map(|value| value.state.open_threads.len())
            .unwrap_or(0);

        json!({
            "schema": "NOLANE-PRODUCT-RUNTIME-STATUS-V1",
            "phase": if self.powered { "on" } else { "off" },
            "powered": self.powered,
            "error": Value::Null,
            "identity_id": identity_id,
            "state_version": self.meta.state_version,
            "interactions": interactions,
            "open_threads": open_threads,
            "memory_enabled": self.meta.memory_enabled,
            "initiative": &self.meta.initiative,
            "model_checkpoint_sha256": self.runtime.source_checkpoint_sha256(),
            "device": "android-local-rust",
            "readiness": {
                "status": "PASS",
                "critical_failures": 0,
                "advisory_failures": 0,
            },
        })
    }

    fn profile_json(&self) -> Result<Value, RuntimeError> {
        let state = self.runtime.persistent_state().ok_or_else(|| {
            RuntimeError::Invalid(
                "LocalMobile persistent state is not attached".into(),
            )
        })?;
        let profile = &state.profile;
        let body = json!({
            "preferred_name": &profile.preferred_name,
            "language": &profile.language,
            "response_length": &profile.response_length,
            "conversation_style": &profile.conversation_style,
            "initiative": &self.meta.initiative,
            "memory_enabled": self.meta.memory_enabled,
            "personal_instruction": &profile.personal_instruction,
        });
        let digest = sha256_hex(&serde_json::to_vec(&body)?);
        let mut output = body;
        output["digest"] = Value::String(digest);
        Ok(output)
    }

    fn update_profile(&mut self, patch: Value) -> Result<Value, RuntimeError> {
        let object = patch.as_object().ok_or_else(|| RuntimeError::Invalid(
            "profile update body must be an object".into(),
        ))?;
        let mut state = self
            .runtime
            .persistent_state()
            .cloned()
            .ok_or_else(|| RuntimeError::Invalid(
                "LocalMobile persistent state is not attached".into(),
            ))?;

        if let Some(value) = object.get("preferred_name").and_then(Value::as_str) {
            state.profile.preferred_name =
                truncate_chars(value, MAX_PROFILE_NAME_CHARS);
        }
        if let Some(value) = object.get("language").and_then(Value::as_str) {
            state.profile.language = value.to_string();
        }
        if let Some(value) = object.get("response_length").and_then(Value::as_str) {
            state.profile.response_length = value.to_string();
        }
        if let Some(value) = object
            .get("conversation_style")
            .and_then(Value::as_str)
        {
            state.profile.conversation_style = value.to_string();
        }
        if let Some(value) = object
            .get("personal_instruction")
            .and_then(Value::as_str)
        {
            state.profile.personal_instruction =
                truncate_chars(value, MAX_PERSONAL_INSTRUCTION_CHARS);
        }
        if let Some(value) = object.get("memory_enabled").and_then(Value::as_bool) {
            self.meta.memory_enabled = value;
        }
        if let Some(value) = object.get("initiative").and_then(Value::as_str) {
            if !matches!(value, "off" | "gentle" | "active") {
                return Err(RuntimeError::Invalid(
                    "unsupported initiative value".into(),
                ));
            }
            self.meta.initiative = value.to_string();
        }

        self.runtime.set_persistent_state(state)?;
        self.runtime.save_persistent_state(&self.state_path)?;
        write_json(&self.meta_path, &self.meta)?;
        self.profile_json()
    }

    fn send_message(&mut self, text: &str) -> Result<Value, RuntimeError> {
        self.send_message_with_token_limit(text, None)
    }

    pub fn synthetic_court_chat(
        &mut self,
        text: &str,
        max_new_tokens: usize,
    ) -> Result<Value, RuntimeError> {
        if max_new_tokens == 0 || max_new_tokens > 16 {
            return Err(RuntimeError::Invalid(
                "synthetic court token limit must be in 1..=16".into(),
            ));
        }
        self.send_message_with_token_limit(text, Some(max_new_tokens))
    }

    fn send_message_with_token_limit(
        &mut self,
        text: &str,
        max_new_tokens: Option<usize>,
    ) -> Result<Value, RuntimeError> {
        if !self.powered {
            return Err(RuntimeError::Invalid(
                "Nolane AI is not running".into(),
            ));
        }
        let clean = text.trim();
        if clean.is_empty() {
            return Err(RuntimeError::Invalid("message is empty".into()));
        }
        if clean.chars().count() > 12_000 {
            return Err(RuntimeError::Invalid("message is too long".into()));
        }

        let mut rng = OsRng;
        let seed = rng.next_u64();
        let generated = match max_new_tokens {
            Some(limit) => {
                let payload = self.runtime.persistent_product_payload(
                    "reply",
                    "conversation",
                    Some(clean),
                )?;
                let prompt = self.runtime.render_product_prompt(&payload)?;
                self.runtime.generate_seeded(
                    &prompt,
                    limit,
                    seed,
                    PRODUCT_SAMPLING_TEMPERATURE,
                    PRODUCT_SAMPLING_TOP_P,
                )?
            }
            None => self.runtime.generate_persistent_product_seeded(
                "reply",
                "conversation",
                Some(clean),
                seed,
            )?,
        };
        let reply = generated.text.trim().to_string();

        let ordinal = self.meta.state_version.saturating_add(1);
        let marker = now_marker();
        self.history.push(LocalMobileMessage {
            event_id: format!("local-mobile-u-{ordinal}"),
            at: marker.clone(),
            role: "user".into(),
            text: clean.to_string(),
        });
        self.history.push(LocalMobileMessage {
            event_id: format!("local-mobile-a-{ordinal}"),
            at: marker,
            role: "assistant".into(),
            text: reply.clone(),
        });
        if self.history.len() > MAX_HISTORY_MESSAGES {
            let excess = self.history.len() - MAX_HISTORY_MESSAGES;
            self.history.drain(0..excess);
        }

        let mut state: PersistentMobileState = self
            .runtime
            .persistent_state()
            .cloned()
            .ok_or_else(|| RuntimeError::Invalid(
                "LocalMobile persistent state disappeared".into(),
            ))?;
        state.state.relationship.interaction_count = state
            .state
            .relationship
            .interaction_count
            .saturating_add(1);
        self.runtime.set_persistent_state(state)?;
        self.runtime.save_persistent_state(&self.state_path)?;

        self.meta.state_version = ordinal;
        write_json(&self.meta_path, &self.meta)?;
        write_json(&self.history_path, &self.history)?;

        Ok(json!({
            "reply": reply,
            "state_version": self.meta.state_version,
            "observer_error": Value::Null,
        }))
    }
}
