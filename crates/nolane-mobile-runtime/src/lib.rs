use nolane_mobile_kernel::{KernelError, MobileKernel};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use thiserror::Error;
use tokenizers::Tokenizer;

pub const PROMPT_CONTRACT_SCHEMA: &str =
    "NOLANE-V052-FROZEN-PRODUCT-PROMPT-CONTRACT-V1";
pub const PROMPT_CONTRACT_AUTHORITY: &str =
    "PROMPT_RENDER_CONTRACT_ONLY_NO_MODEL_AUTHORITY";

pub const PRODUCT_PAYLOAD_SCHEMA: &str =
    "NOLANE-V053-PRODUCT-PAYLOAD-INPUT-V1";
pub const PRODUCT_SYSTEM_PROMPT: &str = "You are the language cortex of Nolane AI Personal.\nYou are not a generic assistant. Speak like a persistent personal companion whose state and memories are supplied by the runtime.\nUse natural language, usually concise. Vietnamese and English are both allowed; follow the user's language.\nYou may disagree, tease gently, joke, or sound mildly annoyed when context supports it, but never guilt the user for leaving, demand attention, threaten abandonment, or claim suffering to pressure them.\nDo not invent memories. Do not claim certainty about the user's emotion; phrase uncertain impressions naturally.\nThe runtime may ask you to initiate a conversation. In that case, do not mention that you were triggered or scored by a policy.\n";

pub const MAX_PRODUCT_OPEN_THREADS: usize = 4;
pub const MAX_PRODUCT_MEMORIES: usize = 8;

pub const SEEDED_SAMPLER_SCHEMA: &str =
    "NOLANE-V054-SEEDED-Q32-NUCLEUS-V1";
pub const PRODUCT_SAMPLING_TEMPERATURE: f64 = 0.78;
pub const PRODUCT_SAMPLING_TOP_P: f64 = 0.90;

const SAMPLER_LOGIT_SCALE: f64 = 1_000.0;
const SAMPLER_EXP_WEIGHT_SCALE: u64 = 1u64 << 40;
const SAMPLER_PROBABILITY_SCALE: u64 = 1u64 << 32;
const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_MUL1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_MUL2: u64 = 0x94D049BB133111EB;

#[derive(Clone, Debug, Deserialize)]
pub struct ProductPayloadProfile {
    pub preferred_name: String,
    pub language: String,
    pub response_length: String,
    pub conversation_style: String,
    pub personal_instruction: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProductPayloadRelationship {
    pub closeness: f64,
    pub trust: f64,
    pub familiarity: f64,
    pub interaction_count: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProductPayloadAffect {
    pub valence: f64,
    pub energy: f64,
    pub playfulness: f64,
    pub concern: f64,
    pub irritation: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProductPayloadState {
    pub identity_id: String,
    pub relationship: ProductPayloadRelationship,
    pub affect: ProductPayloadAffect,
    pub open_threads: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProductPayloadInput {
    pub schema: String,
    pub profile: ProductPayloadProfile,
    pub state: ProductPayloadState,
    pub mode: String,
    pub intent: String,
    pub user_text: Option<String>,
    pub memories: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("kernel error: {0}")]
    Kernel(#[from] KernelError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("{0}")]
    Invalid(String),
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub stopped_on_eos: bool,
    pub final_state: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct PromptSegments {
    prefix: String,
    between: String,
    suffix: String,
}

#[derive(Clone, Debug, Deserialize)]
struct PromptProbe {
    system_sentinel: String,
    user_sentinel: String,
    rendered_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FrozenPromptContract {
    schema: String,
    authority: String,
    tokenizer_json_sha256: String,
    tokenizer_config_json_sha256: String,
    roles: Vec<String>,
    add_generation_prompt: bool,
    enable_thinking: bool,
    segments: PromptSegments,
    probe: PromptProbe,
    contract_sha256: String,
    #[serde(skip)]
    file_sha256: String,
}

pub const MAX_PROMPT_TOKENS: usize = 8192;
pub const MAX_NEW_TOKENS: usize = 512;

pub struct MobileRuntime {
    kernel: MobileKernel,
    tokenizer: Tokenizer,
    latent: Vec<f32>,
    prompt_contract: Option<FrozenPromptContract>,
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

fn sha256_file(path: &Path) -> Result<String, RuntimeError> {
    Ok(sha256_hex(&fs::read(path)?))
}

impl FrozenPromptContract {
    pub fn load(
        contract_json: impl AsRef<Path>,
        tokenizer_json: impl AsRef<Path>,
        tokenizer_config_json: impl AsRef<Path>,
        expected_contract_file_sha256: &str,
    ) -> Result<Self, RuntimeError> {
        let contract_path = contract_json.as_ref();
        let bytes = fs::read(contract_path)?;
        let file_sha256 = sha256_hex(&bytes);
        if file_sha256 != expected_contract_file_sha256.to_ascii_lowercase() {
            return Err(RuntimeError::Invalid(
                "product prompt contract file SHA-256 mismatch".into(),
            ));
        }
        let mut contract: Self = serde_json::from_slice(&bytes)?;
        if contract.schema != PROMPT_CONTRACT_SCHEMA {
            return Err(RuntimeError::Invalid(
                "product prompt contract schema mismatch".into(),
            ));
        }
        if contract.authority != PROMPT_CONTRACT_AUTHORITY {
            return Err(RuntimeError::Invalid(
                "product prompt contract authority mismatch".into(),
            ));
        }
        if contract.roles != ["system".to_string(), "user".to_string()] {
            return Err(RuntimeError::Invalid(
                "product prompt contract role sequence mismatch".into(),
            ));
        }
        if !contract.add_generation_prompt || contract.enable_thinking {
            return Err(RuntimeError::Invalid(
                "product prompt contract generation flags mismatch".into(),
            ));
        }
        if contract.contract_sha256.len() != 64 {
            return Err(RuntimeError::Invalid(
                "product prompt contract digest is malformed".into(),
            ));
        }

        let tokenizer_sha = sha256_file(tokenizer_json.as_ref())?;
        if tokenizer_sha != contract.tokenizer_json_sha256 {
            return Err(RuntimeError::Invalid(
                "prompt contract tokenizer.json SHA-256 mismatch".into(),
            ));
        }
        let tokenizer_config_sha =
            sha256_file(tokenizer_config_json.as_ref())?;
        if tokenizer_config_sha != contract.tokenizer_config_json_sha256 {
            return Err(RuntimeError::Invalid(
                "prompt contract tokenizer_config.json SHA-256 mismatch".into(),
            ));
        }

        let probe = contract.render(
            &contract.probe.system_sentinel,
            &contract.probe.user_sentinel,
        );
        if sha256_hex(probe.as_bytes()) != contract.probe.rendered_sha256 {
            return Err(RuntimeError::Invalid(
                "product prompt contract probe mismatch".into(),
            ));
        }
        contract.file_sha256 = file_sha256;
        Ok(contract)
    }

    pub fn render(&self, system_text: &str, user_text: &str) -> String {
        let mut rendered = String::with_capacity(
            self.segments.prefix.len()
                + system_text.len()
                + self.segments.between.len()
                + user_text.len()
                + self.segments.suffix.len(),
        );
        rendered.push_str(&self.segments.prefix);
        rendered.push_str(system_text);
        rendered.push_str(&self.segments.between);
        rendered.push_str(user_text);
        rendered.push_str(&self.segments.suffix);
        rendered
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
}

fn language_guidance(language: &str) -> Result<&'static str, RuntimeError> {
    match language {
        "auto" => Ok("Follow the user's current language naturally."),
        "vi" => Ok("Prefer Vietnamese unless the user explicitly asks for another language."),
        "en" => Ok("Prefer English unless the user explicitly asks for another language."),
        other => Err(RuntimeError::Invalid(format!(
            "unsupported product language: {other}"
        ))),
    }
}

fn style_guidance(style: &str) -> Result<&'static str, RuntimeError> {
    match style {
        "natural" => Ok("Speak naturally. Avoid canned assistant phrasing."),
        "warm" => Ok("Be warm and attentive without becoming sentimental or clingy."),
        "direct" => Ok("Be direct, concrete and low-fluff."),
        "playful" => Ok("Allow light wit and playfulness when context supports it."),
        other => Err(RuntimeError::Invalid(format!(
            "unsupported conversation style: {other}"
        ))),
    }
}

impl ProductPayloadInput {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.schema != PRODUCT_PAYLOAD_SCHEMA {
            return Err(RuntimeError::Invalid(
                "product payload schema mismatch".into(),
            ));
        }
        if self.state.open_threads.len() > MAX_PRODUCT_OPEN_THREADS {
            return Err(RuntimeError::Invalid(
                "product payload exceeds open-thread limit".into(),
            ));
        }
        if self.memories.len() > MAX_PRODUCT_MEMORIES {
            return Err(RuntimeError::Invalid(
                "product payload exceeds memory limit".into(),
            ));
        }
        if self.mode != "reply" && self.mode != "initiative" {
            return Err(RuntimeError::Invalid(format!(
                "unsupported product payload mode: {}",
                self.mode
            )));
        }
        if self.mode == "initiative"
            && self.user_text.as_deref().unwrap_or("") != ""
        {
            return Err(RuntimeError::Invalid(
                "initiative product payload must not contain user_text".into(),
            ));
        }
        let _ = language_guidance(&self.profile.language)?;
        let _ = style_guidance(&self.profile.conversation_style)?;
        match self.profile.response_length.as_str() {
            "compact" | "balanced" | "expansive" => {}
            other => {
                return Err(RuntimeError::Invalid(format!(
                    "unsupported response length: {other}"
                )))
            }
        }
        let numeric = [
            self.state.relationship.closeness,
            self.state.relationship.trust,
            self.state.relationship.familiarity,
            self.state.affect.valence,
            self.state.affect.energy,
            self.state.affect.playfulness,
            self.state.affect.concern,
            self.state.affect.irritation,
        ];
        if numeric.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::Invalid(
                "product payload contains non-finite state value".into(),
            ));
        }
        Ok(())
    }

    pub fn max_new_tokens(&self) -> Result<usize, RuntimeError> {
        self.validate()?;
        match self.profile.response_length.as_str() {
            "compact" => Ok(96),
            "balanced" => Ok(160),
            "expansive" => Ok(256),
            _ => unreachable!("validated response length"),
        }
    }

    pub fn render_user_payload(&self) -> Result<String, RuntimeError> {
        self.validate()?;

        let preferred_name = if self.profile.preferred_name.is_empty() {
            "(not set)"
        } else {
            self.profile.preferred_name.as_str()
        };
        let personal_instruction = if self.profile.personal_instruction.is_empty() {
            "(none)"
        } else {
            self.profile.personal_instruction.as_str()
        };
        let threads_json = serde_json::to_string(&self.state.open_threads)?;
        let memories = if self.memories.is_empty() {
            "(none)".to_string()
        } else {
            self.memories
                .iter()
                .map(|memory| format!("- {memory}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let task = if self.mode == "reply" {
            format!(
                "User message:\n{}\n\nReply as this persistent personal companion.",
                self.user_text.as_deref().unwrap_or("")
            )
        } else {
            "Initiate one natural, non-intrusive message that genuinely uses the supplied state or open thread.".to_string()
        };

        Ok(format!(
            "Personalization:\npreferred_name={preferred_name}\nlanguage={}: {}\nresponse_length={}\nconversation_style={}: {}\npersonal_instruction={personal_instruction}\nDo not mention these settings unless they are directly relevant.\n\nRuntime state:\nidentity_id={}\nrelationship: closeness={:.2}, trust={:.2}, familiarity={:.2}, interactions={}\nbehavior: valence={:.2}, energy={:.2}, playfulness={:.2}, concern={:.2}, irritation={:.2}\nopen_threads={}\nrequested_intent={}\n\nRelevant memories:\n{}\n\n{}",
            self.profile.language,
            language_guidance(&self.profile.language)?,
            self.profile.response_length,
            self.profile.conversation_style,
            style_guidance(&self.profile.conversation_style)?,
            self.state.identity_id,
            self.state.relationship.closeness,
            self.state.relationship.trust,
            self.state.relationship.familiarity,
            self.state.relationship.interaction_count,
            self.state.affect.valence,
            self.state.affect.energy,
            self.state.affect.playfulness,
            self.state.affect.concern,
            self.state.affect.irritation,
            threads_json,
            self.intent,
            memories,
            task,
        ))
    }
}

fn round_half_up_positive(value: f64) -> Result<u64, RuntimeError> {
    if !value.is_finite() || value < 0.0 {
        return Err(RuntimeError::Invalid(
            "sampler quantization input must be finite and non-negative".into(),
        ));
    }
    Ok((value + 0.5).floor() as u64)
}

fn round_half_away_from_zero(value: f64) -> Result<i64, RuntimeError> {
    if !value.is_finite() {
        return Err(RuntimeError::Invalid(
            "sampler logit must be finite".into(),
        ));
    }
    if value >= 0.0 {
        Ok((value + 0.5).floor() as i64)
    } else {
        Ok((value - 0.5).ceil() as i64)
    }
}

fn splitmix64_next(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut value = state;
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_MUL1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_MUL2);
    value ^= value >> 31;
    (state, value)
}

#[derive(Clone, Debug)]
pub struct SeededNucleusSampler {
    state: u64,
    temperature: f64,
    top_p: f64,
}

impl SeededNucleusSampler {
    pub fn new(
        seed: u64,
        temperature: f64,
        top_p: f64,
    ) -> Result<Self, RuntimeError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(RuntimeError::Invalid(
                "temperature must be finite and positive".into(),
            ));
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            return Err(RuntimeError::Invalid(
                "top_p must be finite and in (0, 1]".into(),
            ));
        }
        Ok(Self {
            state: seed,
            temperature,
            top_p,
        })
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize, RuntimeError> {
        if logits.is_empty() {
            return Err(RuntimeError::Invalid(
                "seeded sampler requires non-empty logits".into(),
            ));
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::Invalid(
                "logits contain non-finite value".into(),
            ));
        }

        let mut quantized_logits = Vec::with_capacity(logits.len());
        for value in logits {
            quantized_logits.push(round_half_away_from_zero(
                *value as f64 * SAMPLER_LOGIT_SCALE,
            )?);
        }
        let maximum = *quantized_logits.iter().max().ok_or_else(|| {
            RuntimeError::Invalid("seeded sampler has no logits".into())
        })?;

        let mut exp_weights = Vec::with_capacity(logits.len());
        let mut total_exp: u64 = 0;
        for value in quantized_logits {
            let delta = ((value - maximum) as f64 / SAMPLER_LOGIT_SCALE)
                / self.temperature;
            let weight = round_half_up_positive(
                delta.exp() * SAMPLER_EXP_WEIGHT_SCALE as f64,
            )?;
            total_exp = total_exp.checked_add(weight).ok_or_else(|| {
                RuntimeError::Invalid(
                    "sampler exponential mass overflow".into(),
                )
            })?;
            exp_weights.push(weight);
        }
        if total_exp == 0 {
            return Err(RuntimeError::Invalid(
                "softmax normalization failed".into(),
            ));
        }

        let mut weights = Vec::with_capacity(exp_weights.len());
        for value in exp_weights {
            let numerator = value as u128
                * SAMPLER_PROBABILITY_SCALE as u128
                + (total_exp / 2) as u128;
            weights.push(
                (numerator / total_exp as u128) as u64
            );
        }
        if !weights.iter().any(|value| *value > 0) {
            return Err(RuntimeError::Invalid(
                "quantized probability mass is empty".into(),
            ));
        }

        let mut ranked: Vec<usize> = (0..weights.len()).collect();
        ranked.sort_by(|left, right| {
            weights[*right]
                .cmp(&weights[*left])
                .then_with(|| left.cmp(right))
        });

        let total: u64 = weights.iter().try_fold(
            0u64,
            |acc, value| acc.checked_add(*value),
        ).ok_or_else(|| {
            RuntimeError::Invalid(
                "quantized probability mass overflow".into(),
            )
        })?;
        let top_p_q32 = round_half_up_positive(
            self.top_p * SAMPLER_PROBABILITY_SCALE as f64,
        )?.min(SAMPLER_PROBABILITY_SCALE);
        let scale = SAMPLER_PROBABILITY_SCALE as u128;
        let threshold = (
            total as u128 * top_p_q32 as u128
            + scale - 1
        ) / scale;
        let threshold = threshold
            .max(1)
            .min(total as u128) as u64;

        let mut retained = Vec::new();
        let mut cumulative = 0u64;
        for token_id in ranked {
            let weight = weights[token_id];
            if weight == 0 {
                continue;
            }
            retained.push((token_id, weight));
            cumulative = cumulative
                .checked_add(weight)
                .ok_or_else(|| RuntimeError::Invalid(
                    "top-p cumulative mass overflow".into(),
                ))?;
            if cumulative >= threshold {
                break;
            }
        }
        if retained.is_empty() {
            return Err(RuntimeError::Invalid(
                "top-p filter removed all probability mass".into(),
            ));
        }

        let retained_total: u64 = retained.iter().try_fold(
            0u64,
            |acc, (_, weight)| acc.checked_add(*weight),
        ).ok_or_else(|| RuntimeError::Invalid(
            "retained probability mass overflow".into(),
        ))?;
        let (state, random_value) = splitmix64_next(self.state);
        self.state = state;
        let draw = random_value % retained_total;

        let mut cumulative = 0u64;
        for (token_id, weight) in retained {
            cumulative = cumulative.checked_add(weight).ok_or_else(|| {
                RuntimeError::Invalid(
                    "sampler draw cumulative overflow".into(),
                )
            })?;
            if draw < cumulative {
                return Ok(token_id);
            }
        }
        Err(RuntimeError::Invalid(
            "seeded sampler draw escaped cumulative mass".into(),
        ))
    }
}

fn argmax(values: &[f32]) -> Result<usize, RuntimeError> {
    if values.is_empty() {
        return Err(RuntimeError::Invalid("cannot argmax empty logits".into()));
    }
    let mut best_index = 0usize;
    let mut best_value = values[0];
    if !best_value.is_finite() {
        return Err(RuntimeError::Invalid("logits contain non-finite value".into()));
    }
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(RuntimeError::Invalid("logits contain non-finite value".into()));
        }
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    Ok(best_index)
}

impl MobileRuntime {
    pub fn load(
        package_dir: impl AsRef<Path>,
        tokenizer_json: impl AsRef<Path>,
        expected_source_checkpoint_sha256: Option<&str>,
        latent: Vec<f32>,
    ) -> Result<Self, RuntimeError> {
        let kernel = MobileKernel::load(
            package_dir,
            expected_source_checkpoint_sha256,
        )?;
        if latent.len() != kernel.contract().latent_dim {
            return Err(RuntimeError::Invalid(format!(
                "latent shape mismatch: expected {}, got {}",
                kernel.contract().latent_dim,
                latent.len(),
            )));
        }
        if latent.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::Invalid(
                "latent contains non-finite value".into(),
            ));
        }

        let tokenizer = Tokenizer::from_file(tokenizer_json.as_ref())
            .map_err(|error| RuntimeError::Tokenizer(error.to_string()))?;
        let tokenizer_vocab = tokenizer.get_vocab_size(true);
        if tokenizer_vocab != kernel.contract().vocab_size {
            return Err(RuntimeError::Invalid(format!(
                "tokenizer/model vocabulary mismatch: tokenizer={} model={}",
                tokenizer_vocab,
                kernel.contract().vocab_size,
            )));
        }

        Ok(Self {
            kernel,
            tokenizer,
            latent,
            prompt_contract: None,
        })
    }

    pub fn load_with_prompt_contract(
        package_dir: impl AsRef<Path>,
        tokenizer_json: impl AsRef<Path>,
        tokenizer_config_json: impl AsRef<Path>,
        prompt_contract_json: impl AsRef<Path>,
        expected_prompt_contract_file_sha256: &str,
        expected_source_checkpoint_sha256: Option<&str>,
        latent: Vec<f32>,
    ) -> Result<Self, RuntimeError> {
        let tokenizer_json_path = tokenizer_json.as_ref();
        let prompt_contract = FrozenPromptContract::load(
            prompt_contract_json,
            tokenizer_json_path,
            tokenizer_config_json,
            expected_prompt_contract_file_sha256,
        )?;
        let mut runtime = Self::load(
            package_dir,
            tokenizer_json_path,
            expected_source_checkpoint_sha256,
            latent,
        )?;
        runtime.prompt_contract = Some(prompt_contract);
        Ok(runtime)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, RuntimeError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| RuntimeError::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            return Err(RuntimeError::Invalid(
                "tokenizer produced an empty prompt".into(),
            ));
        }
        if ids.len() > MAX_PROMPT_TOKENS {
            return Err(RuntimeError::Invalid(format!(
                "prompt exceeds native token limit: {} > {}",
                ids.len(),
                MAX_PROMPT_TOKENS,
            )));
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, RuntimeError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|error| RuntimeError::Tokenizer(error.to_string()))
    }

    pub fn render_chat_prompt(
        &self,
        system_text: &str,
        user_text: &str,
    ) -> Result<String, RuntimeError> {
        let contract = self.prompt_contract.as_ref().ok_or_else(|| {
            RuntimeError::Invalid(
                "native product prompt contract is not loaded".into(),
            )
        })?;
        Ok(contract.render(system_text, user_text))
    }

    pub fn render_product_prompt(
        &self,
        payload: &ProductPayloadInput,
    ) -> Result<String, RuntimeError> {
        let user_payload = payload.render_user_payload()?;
        self.render_chat_prompt(PRODUCT_SYSTEM_PROMPT, &user_payload)
    }

    pub fn generate_product_greedy(
        &self,
        payload: &ProductPayloadInput,
    ) -> Result<GenerationResult, RuntimeError> {
        let max_new_tokens = payload.max_new_tokens()?;
        let prompt = self.render_product_prompt(payload)?;
        self.generate_greedy(&prompt, max_new_tokens)
    }

    pub fn generate_product_seeded(
        &self,
        payload: &ProductPayloadInput,
        seed: u64,
    ) -> Result<GenerationResult, RuntimeError> {
        let max_new_tokens = payload.max_new_tokens()?;
        let prompt = self.render_product_prompt(payload)?;
        self.generate_seeded(
            &prompt,
            max_new_tokens,
            seed,
            PRODUCT_SAMPLING_TEMPERATURE,
            PRODUCT_SAMPLING_TOP_P,
        )
    }

    pub fn generate_chat_greedy(
        &self,
        system_text: &str,
        user_text: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationResult, RuntimeError> {
        let prompt = self.render_chat_prompt(system_text, user_text)?;
        self.generate_greedy(&prompt, max_new_tokens)
    }

    pub fn generate_seeded(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        seed: u64,
        temperature: f64,
        top_p: f64,
    ) -> Result<GenerationResult, RuntimeError> {
        if max_new_tokens == 0 {
            return Err(RuntimeError::Invalid(
                "max_new_tokens must be positive".into(),
            ));
        }
        if max_new_tokens > MAX_NEW_TOKENS {
            return Err(RuntimeError::Invalid(format!(
                "max_new_tokens exceeds native limit: {} > {}",
                max_new_tokens,
                MAX_NEW_TOKENS,
            )));
        }
        let prompt_token_ids = self.encode(prompt)?;
        let mut state = self.kernel.init_state(&self.latent)?;
        let mut next_logits = None;

        for token_id in &prompt_token_ids {
            let output = self.kernel.step(
                *token_id as usize,
                &state,
                &self.latent,
            )?;
            state = output.state;
            next_logits = Some(output.logits);
        }

        let eos = self
            .kernel
            .contract()
            .eos_token_id
            .and_then(|value| usize::try_from(value).ok());
        let mut logits = next_logits.ok_or_else(|| {
            RuntimeError::Invalid("prompt produced no logits".into())
        })?;
        let mut sampler = SeededNucleusSampler::new(
            seed,
            temperature,
            top_p,
        )?;
        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut stopped_on_eos = false;

        for _ in 0..max_new_tokens {
            let token = sampler.sample(&logits)?;
            if token > u32::MAX as usize {
                return Err(RuntimeError::Invalid(
                    "generated token id exceeds u32".into(),
                ));
            }
            generated_token_ids.push(token as u32);
            if eos == Some(token) {
                stopped_on_eos = true;
                break;
            }
            let output = self.kernel.step(
                token,
                &state,
                &self.latent,
            )?;
            state = output.state;
            logits = output.logits;
        }

        let text = self.decode(&generated_token_ids)?;
        Ok(GenerationResult {
            prompt_token_ids,
            generated_token_ids,
            text,
            stopped_on_eos,
            final_state: state,
        })
    }

    pub fn generate_greedy(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationResult, RuntimeError> {
        if max_new_tokens == 0 {
            return Err(RuntimeError::Invalid(
                "max_new_tokens must be positive".into(),
            ));
        }
        if max_new_tokens > MAX_NEW_TOKENS {
            return Err(RuntimeError::Invalid(format!(
                "max_new_tokens exceeds native limit: {} > {}",
                max_new_tokens,
                MAX_NEW_TOKENS,
            )));
        }
        let prompt_token_ids = self.encode(prompt)?;
        let mut state = self.kernel.init_state(&self.latent)?;
        let mut next_logits = None;

        for token_id in &prompt_token_ids {
            let output = self.kernel.step(
                *token_id as usize,
                &state,
                &self.latent,
            )?;
            state = output.state;
            next_logits = Some(output.logits);
        }

        let eos = self
            .kernel
            .contract()
            .eos_token_id
            .and_then(|value| usize::try_from(value).ok());
        let mut logits = next_logits.ok_or_else(|| {
            RuntimeError::Invalid("prompt produced no logits".into())
        })?;
        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut stopped_on_eos = false;

        for _ in 0..max_new_tokens {
            let token = argmax(&logits)?;
            if token > u32::MAX as usize {
                return Err(RuntimeError::Invalid(
                    "generated token id exceeds u32".into(),
                ));
            }
            generated_token_ids.push(token as u32);
            if eos == Some(token) {
                stopped_on_eos = true;
                break;
            }
            let output = self.kernel.step(
                token,
                &state,
                &self.latent,
            )?;
            state = output.state;
            logits = output.logits;
        }

        let text = self.decode(&generated_token_ids)?;
        Ok(GenerationResult {
            prompt_token_ids,
            generated_token_ids,
            text,
            stopped_on_eos,
            final_state: state,
        })
    }

    pub fn source_checkpoint_sha256(&self) -> &str {
        self.kernel.source_checkpoint_sha256()
    }

    pub fn prompt_contract_file_sha256(&self) -> Option<&str> {
        self.prompt_contract
            .as_ref()
            .map(FrozenPromptContract::file_sha256)
    }
}

#[cfg(test)]
mod tests {
    use super::{argmax, sha256_hex, FrozenPromptContract};
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn argmax_is_deterministic_and_first_wins_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
    }

    #[test]
    fn argmax_rejects_non_finite_logits() {
        assert!(argmax(&[1.0, f32::NAN]).is_err());
    }

    #[test]
    fn frozen_prompt_contract_binds_file_and_tokenizer_assets() {
        let root = tempdir().unwrap();
        let tokenizer = root.path().join("tokenizer.json");
        let tokenizer_config = root.path().join("tokenizer_config.json");
        let contract_path = root.path().join("prompt-contract.json");

        fs::write(&tokenizer, b"{\"tokenizer\":\"fixture\"}\n").unwrap();
        fs::write(
            &tokenizer_config,
            b"{\"chat_template\":\"fixture\"}\n",
        )
        .unwrap();

        let system_sentinel = "__SYS__";
        let user_sentinel = "__USR__";
        let rendered_probe =
            format!("<s>{system_sentinel}</s><u>{user_sentinel}</u><a>");
        let payload = json!({
            "schema": "NOLANE-V052-FROZEN-PRODUCT-PROMPT-CONTRACT-V1",
            "authority": "PROMPT_RENDER_CONTRACT_ONLY_NO_MODEL_AUTHORITY",
            "tokenizer_json_sha256": sha256_hex(
                &fs::read(&tokenizer).unwrap()
            ),
            "tokenizer_config_json_sha256": sha256_hex(
                &fs::read(&tokenizer_config).unwrap()
            ),
            "roles": ["system", "user"],
            "add_generation_prompt": true,
            "enable_thinking": false,
            "segments": {
                "prefix": "<s>",
                "between": "</s><u>",
                "suffix": "</u><a>"
            },
            "probe": {
                "system_sentinel": system_sentinel,
                "user_sentinel": user_sentinel,
                "rendered_sha256": sha256_hex(rendered_probe.as_bytes())
            },
            "contract_sha256": "0".repeat(64)
        });
        let bytes = serde_json::to_vec(&payload).unwrap();
        fs::write(&contract_path, &bytes).unwrap();
        let file_sha = sha256_hex(&bytes);

        let contract = FrozenPromptContract::load(
            &contract_path,
            &tokenizer,
            &tokenizer_config,
            &file_sha,
        )
        .unwrap();
        assert_eq!(
            contract.render("hello", "world"),
            "<s>hello</s><u>world</u><a>"
        );

        let wrong_sha = "f".repeat(64);
        let error = FrozenPromptContract::load(
            &contract_path,
            &tokenizer,
            &tokenizer_config,
            &wrong_sha,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("contract file SHA-256 mismatch")
        );

        fs::write(&tokenizer, b"{\"tokenizer\":\"tampered\"}\n").unwrap();
        let error = FrozenPromptContract::load(
            &contract_path,
            &tokenizer,
            &tokenizer_config,
            &file_sha,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("tokenizer.json SHA-256 mismatch")
        );
    }
}
