use crate::{
    ProductPayloadAffect,
    ProductPayloadInput,
    ProductPayloadProfile,
    ProductPayloadRelationship,
    ProductPayloadState,
    RuntimeError,
    MAX_PRODUCT_MEMORIES,
    MAX_PRODUCT_OPEN_THREADS,
    PRODUCT_PAYLOAD_SCHEMA,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

pub const MOBILE_PRODUCT_STATE_SCHEMA: &str =
    "NOLANE-V055-MOBILE-PRODUCT-STATE-V1";
pub const MOBILE_LATENT_SCHEMA: &str =
    "NOLANE-V055-MOBILE-PERSISTENT-LATENT-V1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MobileProfile {
    pub preferred_name: String,
    pub language: String,
    pub response_length: String,
    pub conversation_style: String,
    pub initiative: String,
    pub memory_enabled: bool,
    pub personal_instruction: String,
}

impl Default for MobileProfile {
    fn default() -> Self {
        Self {
            preferred_name: String::new(),
            language: "auto".into(),
            response_length: "balanced".into(),
            conversation_style: "natural".into(),
            initiative: "gentle".into(),
            memory_enabled: true,
            personal_instruction: String::new(),
        }
    }
}

impl MobileProfile {
    pub fn normalize(&mut self) {
        self.preferred_name = trim_to_chars(&self.preferred_name, 80);
        if !matches!(self.language.as_str(), "auto" | "vi" | "en") {
            self.language = "auto".into();
        }
        if !matches!(
            self.response_length.as_str(),
            "compact" | "balanced" | "expansive"
        ) {
            self.response_length = "balanced".into();
        }
        if !matches!(
            self.conversation_style.as_str(),
            "natural" | "warm" | "direct" | "playful"
        ) {
            self.conversation_style = "natural".into();
        }
        if !matches!(self.initiative.as_str(), "off" | "gentle" | "active") {
            self.initiative = "gentle".into();
        }
        self.personal_instruction =
            trim_to_chars(&self.personal_instruction, 1200);
    }

    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.preferred_name.chars().count() > 80 {
            return Err(RuntimeError::Invalid(
                "mobile preferred_name exceeds 80 characters".into(),
            ));
        }
        if !matches!(self.language.as_str(), "auto" | "vi" | "en") {
            return Err(RuntimeError::Invalid(
                "mobile product language is invalid".into(),
            ));
        }
        if !matches!(
            self.response_length.as_str(),
            "compact" | "balanced" | "expansive"
        ) {
            return Err(RuntimeError::Invalid(
                "mobile response_length is invalid".into(),
            ));
        }
        if !matches!(
            self.conversation_style.as_str(),
            "natural" | "warm" | "direct" | "playful"
        ) {
            return Err(RuntimeError::Invalid(
                "mobile conversation_style is invalid".into(),
            ));
        }
        if !matches!(self.initiative.as_str(), "off" | "gentle" | "active") {
            return Err(RuntimeError::Invalid(
                "mobile initiative setting is invalid".into(),
            ));
        }
        if self.personal_instruction.chars().count() > 1200 {
            return Err(RuntimeError::Invalid(
                "mobile personal_instruction exceeds 1200 characters".into(),
            ));
        }
        Ok(())
    }

    pub fn payload_profile(&self) -> ProductPayloadProfile {
        ProductPayloadProfile {
            preferred_name: self.preferred_name.clone(),
            language: self.language.clone(),
            response_length: self.response_length.clone(),
            conversation_style: self.conversation_style.clone(),
            personal_instruction: self.personal_instruction.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MobilePersistentProductState {
    pub schema: String,
    pub profile: MobileProfile,
    pub state: ProductPayloadState,
    pub version: u64,
    pub digest: String,
}

impl MobilePersistentProductState {
    pub fn new(identity_id: impl Into<String>) -> Result<Self, RuntimeError> {
        let identity_id = identity_id.into();
        if identity_id.trim().is_empty() {
            return Err(RuntimeError::Invalid(
                "mobile identity_id cannot be empty".into(),
            ));
        }
        let mut value = Self {
            schema: MOBILE_PRODUCT_STATE_SCHEMA.into(),
            profile: MobileProfile::default(),
            state: ProductPayloadState {
                identity_id,
                relationship: ProductPayloadRelationship {
                    closeness: 0.05,
                    trust: 0.05,
                    familiarity: 0.0,
                    interaction_count: 0,
                },
                affect: ProductPayloadAffect {
                    valence: 0.0,
                    energy: 0.65,
                    playfulness: 0.45,
                    concern: 0.0,
                    irritation: 0.0,
                },
                open_threads: Vec::new(),
            },
            version: 0,
            digest: String::new(),
        };
        value.seal()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.schema != MOBILE_PRODUCT_STATE_SCHEMA {
            return Err(RuntimeError::Invalid(
                "mobile product state schema mismatch".into(),
            ));
        }
        if self.state.identity_id.trim().is_empty()
            || self.state.identity_id.chars().count() > 128
        {
            return Err(RuntimeError::Invalid(
                "mobile product identity_id is invalid".into(),
            ));
        }
        self.profile.validate()?;

        let relationship = &self.state.relationship;
        if relationship.interaction_count < 0
            || !unit(relationship.closeness)
            || !unit(relationship.trust)
            || !unit(relationship.familiarity)
        {
            return Err(RuntimeError::Invalid(
                "mobile relationship state is invalid".into(),
            ));
        }

        let affect = &self.state.affect;
        if !affect.valence.is_finite()
            || !(-1.0..=1.0).contains(&affect.valence)
            || !unit(affect.energy)
            || !unit(affect.playfulness)
            || !unit(affect.concern)
            || !unit(affect.irritation)
        {
            return Err(RuntimeError::Invalid(
                "mobile affect state is invalid".into(),
            ));
        }
        if self.state.open_threads.len() > MAX_PRODUCT_OPEN_THREADS {
            return Err(RuntimeError::Invalid(
                "mobile product state exceeds open-thread limit".into(),
            ));
        }
        if self
            .state
            .open_threads
            .iter()
            .any(|thread| thread.chars().count() > 500)
        {
            return Err(RuntimeError::Invalid(
                "mobile open thread exceeds 500 characters".into(),
            ));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<(), RuntimeError> {
        self.validate()?;
        self.digest = digest_without_field(self, "digest")?;
        Ok(())
    }

    pub fn verify(&self) -> Result<(), RuntimeError> {
        self.validate()?;
        if self.digest.len() != 64
            || digest_without_field(self, "digest")? != self.digest
        {
            return Err(RuntimeError::Invalid(
                "mobile product state digest mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn to_payload(
        &self,
        mode: impl Into<String>,
        intent: impl Into<String>,
        user_text: Option<String>,
        memories: Vec<String>,
    ) -> Result<ProductPayloadInput, RuntimeError> {
        self.verify()?;
        if memories.len() > MAX_PRODUCT_MEMORIES {
            return Err(RuntimeError::Invalid(
                "mobile payload exceeds memory limit".into(),
            ));
        }
        let payload = ProductPayloadInput {
            schema: PRODUCT_PAYLOAD_SCHEMA.into(),
            profile: self.profile.payload_profile(),
            state: self.state.clone(),
            mode: mode.into(),
            intent: intent.into(),
            user_text,
            memories,
        };
        payload.validate()?;
        Ok(payload)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MobilePersistentLatent {
    pub schema: String,
    pub identity_id: String,
    pub checkpoint_sha256: String,
    pub latent_dim: usize,
    pub values: Vec<f32>,
    pub source_state_version: u64,
    pub sequence: u64,
    pub digest: String,
}

impl MobilePersistentLatent {
    pub fn new(
        identity_id: impl Into<String>,
        checkpoint_sha256: impl Into<String>,
        latent_dim: usize,
    ) -> Result<Self, RuntimeError> {
        let mut value = Self {
            schema: MOBILE_LATENT_SCHEMA.into(),
            identity_id: identity_id.into(),
            checkpoint_sha256: checkpoint_sha256.into().to_ascii_lowercase(),
            latent_dim,
            values: vec![0.0; latent_dim],
            source_state_version: 0,
            sequence: 0,
            digest: String::new(),
        };
        value.seal()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.schema != MOBILE_LATENT_SCHEMA {
            return Err(RuntimeError::Invalid(
                "mobile latent schema mismatch".into(),
            ));
        }
        if self.identity_id.trim().is_empty() {
            return Err(RuntimeError::Invalid(
                "mobile latent identity_id cannot be empty".into(),
            ));
        }
        if !is_sha256(&self.checkpoint_sha256) {
            return Err(RuntimeError::Invalid(
                "mobile latent checkpoint SHA-256 is invalid".into(),
            ));
        }
        if self.latent_dim == 0 || self.values.len() != self.latent_dim {
            return Err(RuntimeError::Invalid(
                "mobile latent dimension mismatch".into(),
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::Invalid(
                "mobile latent contains non-finite value".into(),
            ));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<(), RuntimeError> {
        self.validate()?;
        self.digest = digest_without_field(self, "digest")?;
        Ok(())
    }

    pub fn verify(&self) -> Result<(), RuntimeError> {
        self.validate()?;
        if self.digest.len() != 64
            || digest_without_field(self, "digest")? != self.digest
        {
            return Err(RuntimeError::Invalid(
                "mobile latent digest mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn verify_binding(
        &self,
        identity_id: &str,
        checkpoint_sha256: &str,
        latent_dim: usize,
    ) -> Result<(), RuntimeError> {
        self.verify()?;
        if self.identity_id != identity_id {
            return Err(RuntimeError::Invalid(
                "mobile latent identity mismatch".into(),
            ));
        }
        if self.checkpoint_sha256 != checkpoint_sha256.to_ascii_lowercase() {
            return Err(RuntimeError::Invalid(
                "mobile latent checkpoint mismatch".into(),
            ));
        }
        if self.latent_dim != latent_dim {
            return Err(RuntimeError::Invalid(
                "mobile latent dimension mismatch".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MobileStateBundle {
    pub product: MobilePersistentProductState,
    pub latent: MobilePersistentLatent,
    pub latent_reinitialized: bool,
    pub archived_latent: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct MobileStateStore {
    root: PathBuf,
}

impl MobileStateStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn product_state_path(&self) -> PathBuf {
        self.root.join("product-state.json")
    }

    pub fn latent_path(&self) -> PathBuf {
        self.root.join("persistent-latent.json")
    }

    pub fn load_or_initialize(
        &self,
        identity_id: &str,
        checkpoint_sha256: &str,
        latent_dim: usize,
    ) -> Result<MobileStateBundle, RuntimeError> {
        if identity_id.trim().is_empty() {
            return Err(RuntimeError::Invalid(
                "mobile identity_id cannot be empty".into(),
            ));
        }
        if !is_sha256(checkpoint_sha256) {
            return Err(RuntimeError::Invalid(
                "mobile checkpoint SHA-256 is invalid".into(),
            ));
        }
        if latent_dim == 0 {
            return Err(RuntimeError::Invalid(
                "mobile latent_dim must be positive".into(),
            ));
        }

        fs::create_dir_all(&self.root)?;
        let product = match self.load_product_state()? {
            Some(value) => {
                if value.state.identity_id != identity_id {
                    return Err(RuntimeError::Invalid(
                        "mobile product identity mismatch".into(),
                    ));
                }
                value
            }
            None => {
                let value = MobilePersistentProductState::new(identity_id)?;
                self.save_product_state(&value)?;
                value
            }
        };

        let mut latent_reinitialized = false;
        let mut archived_latent = None;
        let latent = match self.load_latent()? {
            Some(value) => {
                if value
                    .verify_binding(
                        identity_id,
                        checkpoint_sha256,
                        latent_dim,
                    )
                    .is_ok()
                {
                    if value.source_state_version > product.version {
                        return Err(RuntimeError::Invalid(
                            "mobile latent source state version is in the future".into(),
                        ));
                    }
                    value
                } else {
                    let archive = self.archive_bound_latent(&value)?;
                    archived_latent = Some(archive);
                    latent_reinitialized = true;
                    let fresh = MobilePersistentLatent::new(
                        identity_id,
                        checkpoint_sha256,
                        latent_dim,
                    )?;
                    self.save_latent(&fresh)?;
                    fresh
                }
            }
            None => {
                let fresh = MobilePersistentLatent::new(
                    identity_id,
                    checkpoint_sha256,
                    latent_dim,
                )?;
                self.save_latent(&fresh)?;
                fresh
            }
        };

        Ok(MobileStateBundle {
            product,
            latent,
            latent_reinitialized,
            archived_latent,
        })
    }

    pub fn update_profile(
        &self,
        mut profile: MobileProfile,
    ) -> Result<MobilePersistentProductState, RuntimeError> {
        profile.normalize();
        profile.validate()?;
        let mut current = self.load_product_state()?.ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile product state is not initialized".into(),
            )
        })?;
        current.profile = profile;
        current.version = current.version.checked_add(1).ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile product state version overflow".into(),
            )
        })?;
        current.seal()?;
        self.save_product_state(&current)?;
        Ok(current)
    }

    pub fn update_product_state(
        &self,
        state: ProductPayloadState,
    ) -> Result<MobilePersistentProductState, RuntimeError> {
        let mut current = self.load_product_state()?.ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile product state is not initialized".into(),
            )
        })?;
        if state.identity_id != current.state.identity_id {
            return Err(RuntimeError::Invalid(
                "mobile product identity cannot change through state update".into(),
            ));
        }
        current.state = state;
        current.version = current.version.checked_add(1).ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile product state version overflow".into(),
            )
        })?;
        current.seal()?;
        self.save_product_state(&current)?;
        Ok(current)
    }

    pub fn update_latent(
        &self,
        values: Vec<f32>,
        source_state_version: u64,
    ) -> Result<MobilePersistentLatent, RuntimeError> {
        let mut current = self.load_latent()?.ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile latent is not initialized".into(),
            )
        })?;
        let product = self.load_product_state()?.ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile product state is not initialized".into(),
            )
        })?;
        if source_state_version != product.version {
            return Err(RuntimeError::Invalid(
                "mobile latent update must bind the current product state version".into(),
            ));
        }
        if current.identity_id != product.state.identity_id {
            return Err(RuntimeError::Invalid(
                "mobile latent/product identity mismatch".into(),
            ));
        }

        if values.len() != current.latent_dim {
            return Err(RuntimeError::Invalid(
                "mobile latent update dimension mismatch".into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::Invalid(
                "mobile latent update contains non-finite value".into(),
            ));
        }
        current.values = values;
        current.source_state_version = source_state_version;
        current.sequence = current.sequence.checked_add(1).ok_or_else(|| {
            RuntimeError::Invalid(
                "mobile latent sequence overflow".into(),
            )
        })?;
        current.seal()?;
        self.save_latent(&current)?;
        Ok(current)
    }

    pub fn load_product_state(
        &self,
    ) -> Result<Option<MobilePersistentProductState>, RuntimeError> {
        load_json_recover_temp(
            &self.product_state_path(),
            MobilePersistentProductState::verify,
        )
    }

    pub fn save_product_state(
        &self,
        value: &MobilePersistentProductState,
    ) -> Result<(), RuntimeError> {
        let mut sealed = value.clone();
        sealed.seal()?;
        atomic_json_write(&self.product_state_path(), &sealed)
    }

    pub fn load_latent(
        &self,
    ) -> Result<Option<MobilePersistentLatent>, RuntimeError> {
        load_json_recover_temp(
            &self.latent_path(),
            MobilePersistentLatent::verify,
        )
    }

    pub fn save_latent(
        &self,
        value: &MobilePersistentLatent,
    ) -> Result<(), RuntimeError> {
        let mut sealed = value.clone();
        sealed.seal()?;
        atomic_json_write(&self.latent_path(), &sealed)
    }

    fn archive_bound_latent(
        &self,
        value: &MobilePersistentLatent,
    ) -> Result<PathBuf, RuntimeError> {
        value.verify()?;
        let path = self.latent_path();
        let archive = self.root.join(format!(
            "persistent-latent.{}.previous.json",
            value.digest,
        ));
        if archive.exists() {
            let existing: MobilePersistentLatent =
                read_verified_json(&archive, MobilePersistentLatent::verify)?;
            if existing.digest != value.digest {
                return Err(RuntimeError::Invalid(
                    "mobile latent archive digest collision".into(),
                ));
            }
            if path.exists() {
                fs::remove_file(&path)?;
            }
            return Ok(archive);
        }
        fs::rename(&path, &archive)?;
        Ok(archive)
    }
}

fn trim_to_chars(value: &str, limit: usize) -> String {
    value.trim().chars().take(limit).collect()
}

fn unit(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn digest_without_field<T: Serialize>(
    value: &T,
    field: &str,
) -> Result<String, RuntimeError> {
    let mut json = serde_json::to_value(value)?;
    let object = json.as_object_mut().ok_or_else(|| {
        RuntimeError::Invalid("persistent payload must be a JSON object".into())
    })?;
    object.remove(field);
    let canonical = canonical_json(&json)?;
    Ok(sha256_hex(canonical.as_bytes()))
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

fn canonical_json(value: &Value) -> Result<String, RuntimeError> {
    let mut out = String::new();
    write_canonical_json(value, &mut out)?;
    Ok(out)
}

fn write_canonical_json(
    value: &Value,
    out: &mut String,
) -> Result<(), RuntimeError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => {
            out.push_str(if *flag { "true" } else { "false" })
        }
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => out.push_str(&serde_json::to_string(text)?),
        Value::Array(values) => {
            out.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_canonical_json(&map[key], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("state.json");
    path.with_file_name(format!(".{name}.tmp"))
}

fn atomic_json_write<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), RuntimeError> {
    let parent = path.parent().ok_or_else(|| {
        RuntimeError::Invalid("persistent path has no parent".into())
    })?;
    fs::create_dir_all(parent)?;
    let temp = temp_path(path);
    let bytes = serde_json::to_vec(value)?;
    let mut handle = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp)?;
    handle.write_all(&bytes)?;
    handle.write_all(b"\n")?;
    handle.sync_all()?;
    drop(handle);
    fs::rename(&temp, path)?;
    if let Ok(directory) = fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn read_verified_json<T, F>(
    path: &Path,
    verify: F,
) -> Result<T, RuntimeError>
where
    T: for<'de> Deserialize<'de>,
    F: Fn(&T) -> Result<(), RuntimeError>,
{
    let value: T = serde_json::from_slice(&fs::read(path)?)?;
    verify(&value)?;
    Ok(value)
}

fn load_json_recover_temp<T, F>(
    path: &Path,
    verify: F,
) -> Result<Option<T>, RuntimeError>
where
    T: for<'de> Deserialize<'de>,
    F: Fn(&T) -> Result<(), RuntimeError> + Copy,
{
    let temp = temp_path(path);
    if path.exists() {
        let value = read_verified_json(path, verify)?;
        if temp.exists() {
            let _ = fs::remove_file(temp);
        }
        return Ok(Some(value));
    }
    if !temp.exists() {
        return Ok(None);
    }

    let value = read_verified_json(&temp, verify)?;
    fs::rename(&temp, path)?;
    Ok(Some(value))
}
