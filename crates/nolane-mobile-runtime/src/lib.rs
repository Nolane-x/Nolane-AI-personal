use nolane_mobile_kernel::{KernelError, MobileKernel};
use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("kernel error: {0}")]
    Kernel(#[from] KernelError),
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

pub struct MobileRuntime {
    kernel: MobileKernel,
    tokenizer: Tokenizer,
    latent: Vec<f32>,
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
        })
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
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, RuntimeError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|error| RuntimeError::Tokenizer(error.to_string()))
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
