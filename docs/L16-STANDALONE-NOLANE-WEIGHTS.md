# L16 Standalone Nolane Weights

## Boundary crossed

L15 removed the Qwen Transformer decoder from inference but still required a loaded Qwen model object to supply token embeddings, final RMS normalization and the language-model head.

L16 exports those remaining inference tensors into a **Nolane-owned standalone checkpoint**.

The standalone runtime path is:

```text
owned token embedding
        |
        v
Nolane deep recurrent state-space cortex
        |
        v
owned RMSNorm
        |
        v
owned output projection
```

No Qwen model object is constructed or accepted by the standalone loader.

## What is copied

The first standalone export intentionally preserves the exact L15 boundary behavior instead of compressing it prematurely:

- token embedding matrix;
- final RMSNorm weight;
- LM-head weight when it is not tied;
- tied embedding/output semantics when the source model shares weights;
- deep recurrent Nolane cortex state.

Boundary dtype is preserved across export/load: FP32, FP16 or BF16.

This makes L16 an **ownership/runtime-independence wave**, not a claim that the input/output language matrices were learned from scratch.

## Export

```bash
python scripts/export_standalone_nolane.py
```

Export requires Qwen and the L15 artifact once. The resulting `standalone-nolane.pt` contains only:

- standalone boundary config;
- owned boundary tensors;
- Nolane cortex config/state;
- source lineage metadata.

It does not contain Qwen Transformer decoder tensors.

## Standalone load

```python
from nolane_personal.standalone_artifact import load_standalone_model

model, metadata = load_standalone_model(
    "runtime-data/l16-standalone/standalone-nolane.pt",
    living_latent,
    device="cpu",
)
```

The loader takes **no Qwen model or Qwen model path** and imports no Transformers model implementation.

At the core model level, runtime requires PyTorch plus the standalone checkpoint.

## Generation

The lowest-level standalone generator consumes token IDs directly:

```bash
python scripts/generate_standalone_nolane.py \
  --input-ids 1,42,108,77
```

Text tokenization remains tokenizer-compatible with the inherited vocabulary. A text frontend may still use an external tokenizer implementation; that is separate from the model-weight runtime boundary.

## Parity court

```bash
python scripts/evaluate_standalone_parity.py
```

The court compares L15 and L16 on the same input IDs and requires:

- tight logit parity;
- tight recurrent-state parity;
- identical greedy generation;
- full-prompt == incremental recurrent scan;
- identical Nolane cortex digest;
- no Qwen model reference on the standalone object;
- no Transformers dependency declared by the core artifact;
- no decoder tensor keys in the checkpoint;
- source-mutation isolation.

The mutation court edits an embedding row that is actually present in the tested prompt. Standalone output must remain unchanged.

## Resource court

```bash
python scripts/benchmark_standalone_resources.py
```

The resource court compares the standalone runtime against L15 and checks checkpoint overhead relative to the tensors that must actually be owned.

## Promotion

```bash
python scripts/decide_standalone_promotion.py \
  --parity runtime-data/l16-parity.json \
  --resources runtime-data/l16-resources.json
```

Parity and resource evidence must reference the same standalone checkpoint and the same source L15 checkpoint.

## Neural court evidence

CI builds a real tiny Qwen3 source model and proves:

- boundary tensors are cloned into distinct storage;
- tied and untied output-weight contracts are preserved;
- exported L16 logits/state match L15;
- greedy generation matches L15;
- standalone prompt scan is recurrently equivalent;
- source Qwen embedding/norm/head mutation cannot alter standalone output;
- the Qwen object can be deleted and garbage-collected while standalone generation still works;
- standalone checkpoint contains no decoder-layer/attention/MLP tensor sections;
- boundary and cortex digests survive artifact roundtrip.

## Scientific boundary

L16 requires **no Qwen model weights at inference other than the tensors that have been explicitly exported into the Nolane checkpoint**.

Those exported boundary matrices still originate from Qwen. The model also still uses token IDs compatible with the inherited tokenizer.

Therefore L16 is standalone in runtime ownership, but not yet independent in learned vocabulary/output representation provenance.

The next compression boundary is to factorize or distill the large embedding/output matrices, optionally followed by a Vietnamese/English-focused native tokenizer, while preserving held-out language quality.
