use safetensors::{tensor::Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path};
use thiserror::Error;

pub const CONTRACT_SCHEMA: &str = "NOLANE-V050-MOBILE-FACTORIZED-TOKEN-STEP-V1";
pub const PACKAGE_SCHEMA: &str = "NOLANE-V050-MOBILE-FACTORIZED-PACKAGE-V1";
pub const PACKAGE_AUTHORITY: &str = "MOBILE_EXPORT_ONLY_NO_PROMOTION_AUTHORITY";

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("{0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Deserialize)]
pub struct MobileContract {
    pub schema: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub rank: usize,
    pub latent_dim: usize,
    pub state_dim: usize,
    pub packed_state_dim: usize,
    pub virtual_steps: usize,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f32,
    pub hidden_norm_eps: f32,
    pub latent_norm_eps: f32,
    pub slow_decay_floor: f32,
    pub max_abs_gate: f32,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
struct MobileManifest {
    schema: String,
    authority: String,
    source_checkpoint_sha256: String,
    contract_filename: String,
    contract_sha256: String,
    weights_filename: String,
    weights_sha256: String,
    tensor_count: usize,
    tensor_names: Vec<String>,
    weights_dtype: String,
}

#[derive(Clone, Debug)]
struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl OwnedTensor {
    fn expect_shape(&self, name: &str, shape: &[usize]) -> Result<(), KernelError> {
        if self.shape != shape {
            return Err(KernelError::Invalid(format!(
                "tensor {name} shape mismatch: expected {shape:?}, got {:?}",
                self.shape
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StepOutput {
    pub logits: Vec<f32>,
    pub state: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MobileKernel {
    contract: MobileContract,
    source_checkpoint_sha256: String,
    tensors: HashMap<String, OwnedTensor>,
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

fn read_f32_tensor(
    name: &str,
    view: safetensors::tensor::TensorView<'_>,
) -> Result<OwnedTensor, KernelError> {
    if view.dtype() != Dtype::F32 {
        return Err(KernelError::Invalid(format!(
            "tensor {name} must be F32, got {:?}",
            view.dtype()
        )));
    }
    let bytes = view.data();
    if bytes.len() % 4 != 0 {
        return Err(KernelError::Invalid(format!(
            "tensor {name} has misaligned F32 payload"
        )));
    }
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let expected: usize = view.shape().iter().product();
    if data.len() != expected {
        return Err(KernelError::Invalid(format!(
            "tensor {name} element count mismatch"
        )));
    }
    Ok(OwnedTensor {
        shape: view.shape().to_vec(),
        data,
    })
}

fn linear(x: &[f32], weight: &OwnedTensor, bias: &OwnedTensor) -> Result<Vec<f32>, KernelError> {
    if weight.shape.len() != 2 {
        return Err(KernelError::Invalid("linear weight must be rank-2".into()));
    }
    let out = weight.shape[0];
    let input = weight.shape[1];
    if x.len() != input {
        return Err(KernelError::Invalid(format!(
            "linear input mismatch: expected {input}, got {}",
            x.len()
        )));
    }
    bias.expect_shape("linear bias", &[out])?;
    let mut y = vec![0.0f32; out];
    for row in 0..out {
        let mut sum = bias.data[row];
        let base = row * input;
        for col in 0..input {
            sum += weight.data[base + col] * x[col];
        }
        y[row] = sum;
    }
    Ok(y)
}

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Result<Vec<f32>, KernelError> {
    if x.len() != weight.len() || x.len() != bias.len() || x.is_empty() {
        return Err(KernelError::Invalid("layer norm shape mismatch".into()));
    }
    let n = x.len() as f32;
    let mean = x.iter().copied().sum::<f32>() / n;
    let var = x
        .iter()
        .map(|value| {
            let d = *value - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let inv = 1.0 / (var + eps).sqrt();
    Ok(x
        .iter()
        .enumerate()
        .map(|(index, value)| (*value - mean) * inv * weight[index] + bias[index])
        .collect())
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, KernelError> {
    if x.len() != weight.len() || x.is_empty() {
        return Err(KernelError::Invalid("RMS norm shape mismatch".into()));
    }
    let mean_sq = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Ok(x
        .iter()
        .enumerate()
        .map(|(index, value)| value * inv * weight[index])
        .collect())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn tanh_vec(values: &mut [f32]) {
    for value in values {
        *value = value.tanh();
    }
}

impl MobileKernel {
    pub fn load(
        package_dir: impl AsRef<Path>,
        expected_source_checkpoint_sha256: Option<&str>,
    ) -> Result<Self, KernelError> {
        let root = package_dir.as_ref();
        let manifest_bytes = fs::read(root.join("manifest.json"))?;
        let manifest: MobileManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.schema != PACKAGE_SCHEMA {
            return Err(KernelError::Invalid("mobile package schema mismatch".into()));
        }
        if manifest.authority != PACKAGE_AUTHORITY {
            return Err(KernelError::Invalid("mobile package authority mismatch".into()));
        }
        if manifest.weights_dtype != "float32" {
            return Err(KernelError::Invalid("mobile package must use float32 weights".into()));
        }
        if let Some(expected) = expected_source_checkpoint_sha256 {
            if manifest.source_checkpoint_sha256 != expected.to_ascii_lowercase() {
                return Err(KernelError::Invalid(
                    "mobile package source checkpoint mismatch".into(),
                ));
            }
        }

        let contract_bytes = fs::read(root.join(&manifest.contract_filename))?;
        if sha256_hex(&contract_bytes) != manifest.contract_sha256 {
            return Err(KernelError::Invalid("mobile contract SHA-256 mismatch".into()));
        }
        let contract: MobileContract = serde_json::from_slice(&contract_bytes)?;
        if contract.schema != CONTRACT_SCHEMA {
            return Err(KernelError::Invalid("mobile contract schema mismatch".into()));
        }
        if contract.packed_state_dim != contract.state_dim * 3 {
            return Err(KernelError::Invalid(
                "packed_state_dim must equal 3 * state_dim".into(),
            ));
        }
        if contract.vocab_size == 0
            || contract.hidden_size == 0
            || contract.rank == 0
            || contract.latent_dim == 0
            || contract.state_dim == 0
            || contract.virtual_steps == 0
        {
            return Err(KernelError::Invalid(
                "mobile contract dimensions must be positive".into(),
            ));
        }

        let weight_bytes = fs::read(root.join(&manifest.weights_filename))?;
        if sha256_hex(&weight_bytes) != manifest.weights_sha256 {
            return Err(KernelError::Invalid("mobile weights SHA-256 mismatch".into()));
        }
        // The Python export verifier checks safetensors __metadata__.
        // Native loading is bound to the immutable weights file by its
        // manifest SHA-256, then independently binds the manifest to the
        // expected source checkpoint and exact tensor-name set. Avoid parsing
        // private safetensors header internals here.
        let safe = SafeTensors::deserialize(&weight_bytes)?;

        let mut tensors = HashMap::new();
        for (name, view) in safe.tensors() {
            tensors.insert(name.clone(), read_f32_tensor(&name, view)?);
        }
        let mut names: Vec<_> = tensors.keys().cloned().collect();
        names.sort();
        let mut manifest_names = manifest.tensor_names.clone();
        manifest_names.sort();
        if names != manifest_names || names.len() != manifest.tensor_count {
            return Err(KernelError::Invalid(
                "mobile tensor manifest mismatch".into(),
            ));
        }

        let kernel = Self {
            contract,
            source_checkpoint_sha256: manifest.source_checkpoint_sha256,
            tensors,
        };
        kernel.validate_tensor_shapes()?;
        Ok(kernel)
    }

    pub fn contract(&self) -> &MobileContract {
        &self.contract
    }

    pub fn source_checkpoint_sha256(&self) -> &str {
        &self.source_checkpoint_sha256
    }

    fn tensor(&self, name: &str) -> Result<&OwnedTensor, KernelError> {
        self.tensors
            .get(name)
            .ok_or_else(|| KernelError::Invalid(format!("missing tensor {name}")))
    }

    fn validate_tensor_shapes(&self) -> Result<(), KernelError> {
        let c = &self.contract;
        let h = c.hidden_size;
        let d = c.state_dim;
        let l = c.latent_dim;
        let v = c.vocab_size;
        let r = c.rank;

        for prefix in ["init", "step"] {
            self.tensor(&format!("{prefix}.latent_norm_weight"))?
                .expect_shape("latent_norm_weight", &[l])?;
            self.tensor(&format!("{prefix}.latent_proj_weight"))?
                .expect_shape("latent_proj_weight", &[d, l])?;
            self.tensor(&format!("{prefix}.latent_proj_bias"))?
                .expect_shape("latent_proj_bias", &[d])?;
        }

        self.tensor("step.input_codes")?
            .expect_shape("input_codes", &[v, r])?;
        self.tensor("step.input_basis")?
            .expect_shape("input_basis", &[r, h])?;
        self.tensor("step.final_norm_weight")?
            .expect_shape("final_norm_weight", &[h])?;
        if !c.tie_word_embeddings {
            self.tensor("step.output_codes")?
                .expect_shape("output_codes", &[v, r])?;
            self.tensor("step.output_basis")?
                .expect_shape("output_basis", &[r, h])?;
        }

        self.tensor("step.hidden_norm_weight")?
            .expect_shape("hidden_norm_weight", &[h])?;
        self.tensor("step.hidden_norm_bias")?
            .expect_shape("hidden_norm_bias", &[h])?;
        self.tensor("step.hidden_down_weight")?
            .expect_shape("hidden_down_weight", &[d, h])?;
        self.tensor("step.hidden_down_bias")?
            .expect_shape("hidden_down_bias", &[d])?;

        for name in [
            "fast_proposal",
            "fast_decay",
            "slow_proposal",
            "slow_decay",
            "fast_output_gate",
            "slow_output_gate",
            "depth_transition",
            "depth_token",
            "depth_gate",
        ] {
            self.tensor(&format!("step.{name}_weight"))?
                .expect_shape(name, &[d, d])?;
            self.tensor(&format!("step.{name}_bias"))?
                .expect_shape(name, &[d])?;
        }
        self.tensor("step.state_out_weight")?
            .expect_shape("state_out_weight", &[h, 3 * d])?;
        self.tensor("step.state_out_bias")?
            .expect_shape("state_out_bias", &[h])?;
        let depth = self.tensor("step.depth_embedding")?;
        if depth.shape.len() != 2
            || depth.shape[1] != d
            || depth.shape[0] < c.virtual_steps
        {
            return Err(KernelError::Invalid(
                "depth_embedding shape is incompatible with contract".into(),
            ));
        }
        if self.tensor("step.raw_gate")?.data.len() != 1 {
            return Err(KernelError::Invalid("raw_gate must be scalar".into()));
        }
        Ok(())
    }

    fn latent_feature(&self, prefix: &str, latent: &[f32]) -> Result<Vec<f32>, KernelError> {
        if latent.len() != self.contract.latent_dim {
            return Err(KernelError::Invalid(format!(
                "latent shape mismatch: expected {}, got {}",
                self.contract.latent_dim,
                latent.len()
            )));
        }
        let norm = self.tensor(&format!("{prefix}.latent_norm_weight"))?;
        let normalized = rms_norm(latent, &norm.data, self.contract.latent_norm_eps)?;
        let weight = self.tensor(&format!("{prefix}.latent_proj_weight"))?;
        let bias = self.tensor(&format!("{prefix}.latent_proj_bias"))?;
        let mut feature = linear(&normalized, weight, bias)?;
        tanh_vec(&mut feature);
        Ok(feature)
    }

    pub fn init_state(&self, latent: &[f32]) -> Result<Vec<f32>, KernelError> {
        let feature = self.latent_feature("init", latent)?;
        let mut state = Vec::with_capacity(self.contract.packed_state_dim);
        state.extend_from_slice(&feature);
        state.extend_from_slice(&feature);
        state.extend_from_slice(&feature);
        Ok(state)
    }

    pub fn step(
        &self,
        token_id: usize,
        state: &[f32],
        latent: &[f32],
    ) -> Result<StepOutput, KernelError> {
        let c = &self.contract;
        if token_id >= c.vocab_size {
            return Err(KernelError::Invalid("token id outside vocabulary".into()));
        }
        if state.len() != c.packed_state_dim {
            return Err(KernelError::Invalid(format!(
                "state shape mismatch: expected {}, got {}",
                c.packed_state_dim,
                state.len()
            )));
        }

        let input_codes = self.tensor("step.input_codes")?;
        let input_basis = self.tensor("step.input_basis")?;
        let code_start = token_id * c.rank;
        let code = &input_codes.data[code_start..code_start + c.rank];
        let mut hidden = vec![0.0f32; c.hidden_size];
        for rank in 0..c.rank {
            let scale = code[rank];
            let base = rank * c.hidden_size;
            for h in 0..c.hidden_size {
                hidden[h] += scale * input_basis.data[base + h];
            }
        }

        let hidden_norm_weight = &self.tensor("step.hidden_norm_weight")?.data;
        let hidden_norm_bias = &self.tensor("step.hidden_norm_bias")?.data;
        let normalized_hidden = layer_norm(
            &hidden,
            hidden_norm_weight,
            hidden_norm_bias,
            c.hidden_norm_eps,
        )?;
        let mut token = linear(
            &normalized_hidden,
            self.tensor("step.hidden_down_weight")?,
            self.tensor("step.hidden_down_bias")?,
        )?;
        tanh_vec(&mut token);
        let latent_feature = self.latent_feature("step", latent)?;

        let d = c.state_dim;
        let mut fast = state[..d].to_vec();
        let mut slow = state[d..2 * d].to_vec();
        let mut depth = state[2 * d..3 * d].to_vec();

        let mut fast_proposal = linear(
            &token,
            self.tensor("step.fast_proposal_weight")?,
            self.tensor("step.fast_proposal_bias")?,
        )?;
        for i in 0..d {
            fast_proposal[i] += latent_feature[i];
        }
        tanh_vec(&mut fast_proposal);
        let fast_decay = linear(
            &token,
            self.tensor("step.fast_decay_weight")?,
            self.tensor("step.fast_decay_bias")?,
        )?
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
        for i in 0..d {
            fast[i] = fast_decay[i] * fast[i]
                + (1.0 - fast_decay[i]) * fast_proposal[i];
        }

        let mut slow_proposal = linear(
            &fast,
            self.tensor("step.slow_proposal_weight")?,
            self.tensor("step.slow_proposal_bias")?,
        )?;
        for i in 0..d {
            slow_proposal[i] += latent_feature[i];
        }
        tanh_vec(&mut slow_proposal);
        let raw_slow_decay = linear(
            &token,
            self.tensor("step.slow_decay_weight")?,
            self.tensor("step.slow_decay_bias")?,
        )?
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
        for i in 0..d {
            let decay = c.slow_decay_floor
                + (1.0 - c.slow_decay_floor) * raw_slow_decay[i];
            slow[i] = decay * slow[i] + (1.0 - decay) * slow_proposal[i];
            depth[i] = 0.5 * depth[i] + 0.25 * fast[i] + 0.25 * slow[i];
        }

        let depth_embedding = self.tensor("step.depth_embedding")?;
        for virtual_step in 0..c.virtual_steps {
            let step_feature = &depth_embedding.data
                [virtual_step * d..(virtual_step + 1) * d];
            let mut proposal = linear(
                &depth,
                self.tensor("step.depth_transition_weight")?,
                self.tensor("step.depth_transition_bias")?,
            )?;
            let depth_token = linear(
                &token,
                self.tensor("step.depth_token_weight")?,
                self.tensor("step.depth_token_bias")?,
            )?;
            for i in 0..d {
                proposal[i] += depth_token[i] + latent_feature[i] + step_feature[i];
            }
            tanh_vec(&mut proposal);

            let depth_plus_step = depth
                .iter()
                .zip(step_feature.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let carry = linear(
                &depth_plus_step,
                self.tensor("step.depth_gate_weight")?,
                self.tensor("step.depth_gate_bias")?,
            )?
            .into_iter()
            .map(sigmoid)
            .collect::<Vec<_>>();
            for i in 0..d {
                depth[i] = carry[i] * depth[i] + (1.0 - carry[i]) * proposal[i];
            }
        }

        let fast_gate = linear(
            &token,
            self.tensor("step.fast_output_gate_weight")?,
            self.tensor("step.fast_output_gate_bias")?,
        )?
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
        let slow_gate = linear(
            &token,
            self.tensor("step.slow_output_gate_weight")?,
            self.tensor("step.slow_output_gate_bias")?,
        )?
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();

        let mut joined = Vec::with_capacity(3 * d);
        for i in 0..d {
            joined.push(fast[i] * fast_gate[i]);
        }
        for i in 0..d {
            joined.push(slow[i] * slow_gate[i]);
        }
        joined.extend_from_slice(&depth);

        let residual = linear(
            &joined,
            self.tensor("step.state_out_weight")?,
            self.tensor("step.state_out_bias")?,
        )?;
        let raw_gate = self.tensor("step.raw_gate")?.data[0];
        let gate = c.max_abs_gate * raw_gate.tanh();
        for i in 0..c.hidden_size {
            hidden[i] += gate * residual[i];
        }

        let final_norm = rms_norm(
            &hidden,
            &self.tensor("step.final_norm_weight")?.data,
            c.rms_norm_eps,
        )?;
        let (output_codes, output_basis) = if c.tie_word_embeddings {
            (
                self.tensor("step.input_codes")?,
                self.tensor("step.input_basis")?,
            )
        } else {
            (
                self.tensor("step.output_codes")?,
                self.tensor("step.output_basis")?,
            )
        };

        let mut rank_hidden = vec![0.0f32; c.rank];
        for rank in 0..c.rank {
            let base = rank * c.hidden_size;
            for h in 0..c.hidden_size {
                rank_hidden[rank] += final_norm[h] * output_basis.data[base + h];
            }
        }
        let mut logits = vec![0.0f32; c.vocab_size];
        for vocab in 0..c.vocab_size {
            let base = vocab * c.rank;
            for rank in 0..c.rank {
                logits[vocab] += rank_hidden[rank] * output_codes.data[base + rank];
            }
        }

        let mut next_state = Vec::with_capacity(c.packed_state_dim);
        next_state.extend_from_slice(&fast);
        next_state.extend_from_slice(&slow);
        next_state.extend_from_slice(&depth);
        Ok(StepOutput {
            logits,
            state: next_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{layer_norm, rms_norm, sigmoid};

    #[test]
    fn scalar_math_is_finite_and_stable() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        let layer = layer_norm(&[1.0, 2.0], &[1.0, 1.0], &[0.0, 0.0], 1e-5)
            .unwrap();
        assert!(layer.iter().all(|v| v.is_finite()));
        let rms = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 1e-6).unwrap();
        assert!(rms.iter().all(|v| v.is_finite()));
    }
}
