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

    pub fn generate_chat_greedy(
        &self,
        system_text: &str,
        user_text: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationResult, RuntimeError> {
        let prompt = self.render_chat_prompt(system_text, user_text)?;
        self.generate_greedy(&prompt, max_new_tokens)
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
    use super::argmax;

    #[test]
    fn argmax_is_deterministic_and_first_wins_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
    }

    #[test]
    fn argmax_rejects_non_finite_logits() {
        assert!(argmax(&[1.0, f32::NAN]).is_err());
    }
}
