from __future__ import annotations

from .store import payload_digest


def build_native_boundary_spec(
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> dict:
    payload = {
        "schema": "NOLANE-L15-NATIVE-BOUNDARY-SPEC-V1",
        "authority": "FROZEN_TRAINING_SPEC_ONLY",
        "base_model_fingerprint": str(base_model_fingerprint),
        "dataset_fingerprint": str(dataset_fingerprint),
        "qwen_components_used": ["embed_tokens", "final_norm", "lm_head"],
        "qwen_decoder_layers_executed": 0,
        "generation_owner": "NOLANE_RECURRENT_STATE",
        "hf_kv_cache_used": False,
        "teacher": "FULL_FROZEN_QWEN",
    }
    payload["spec_sha256"] = payload_digest(payload)
    return payload


def verify_native_boundary_spec(
    spec: dict,
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> None:
    if spec.get("schema") != "NOLANE-L15-NATIVE-BOUNDARY-SPEC-V1":
        raise ValueError("unsupported native-boundary spec")
    supplied = spec.get("spec_sha256")
    body = dict(spec)
    body.pop("spec_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("native-boundary spec digest mismatch")
    if spec.get("base_model_fingerprint") != base_model_fingerprint:
        raise ValueError("native-boundary base-model mismatch")
    if spec.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("native-boundary dataset mismatch")
    if int(spec.get("qwen_decoder_layers_executed", -1)) != 0:
        raise ValueError("native-boundary spec permits Qwen decoder execution")
    if spec.get("hf_kv_cache_used") is not False:
        raise ValueError("native-boundary spec permits HF KV cache")
