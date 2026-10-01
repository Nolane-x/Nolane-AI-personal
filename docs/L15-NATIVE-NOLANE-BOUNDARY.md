# L15 Native Nolane Boundary

## Boundary crossed

L14 still retained two Qwen decoder anchors.

L15 removes the **entire Qwen Transformer decoder stack** from the experimental inference path.

The native path is:

```text
Qwen token embedding
        |
        v
Nolane deep recurrent state-space cortex
        |
        v
Qwen final normalization
        |
        v
Qwen LM head
```

No Qwen attention block or Qwen MLP block executes during native forward or native generation.

## Native autoregressive generation

L15 does not call Hugging Face `generate()` and does not use Transformer KV cache.

The prompt is scanned once by the Nolane cortex. The packed recurrent state is then carried token by token:

```text
prompt embeddings -> cortex scan -> state S
                         |
                         v
                    next-token logits
                         |
new token embedding -> cortex(S) -> S'
                         |
                         v
                    next-token logits
```

This avoids the cache semantic problem that would occur if decoder layer 0 were simply bypassed inside Qwen generation.

## Trainable architecture

The default native cortex is inherited from L14:

- fast temporal recurrent state;
- slow temporal recurrent state;
- virtual-depth recurrent state;
- eight shared virtual-depth micro-steps per token;
- persistent 32D Living latent conditioning;
- hidden-size-1024 footprint: **89,805 trainable parameters**.

Qwen embedding, final norm and LM head are frozen boundary components.

## Frozen boundary specification

```bash
python scripts/freeze_native_boundary_spec.py
```

The spec binds:

- exact Qwen fingerprint;
- exact personalization protocol;
- Qwen components allowed: embedding, final norm, LM head;
- Qwen decoder layers executed: **0**;
- HF KV cache: **false**;
- generation owner: **Nolane recurrent state**.

## Training

```bash
python scripts/train_native_boundary.py
```

Full frozen Qwen acts only as a teacher during training.

For each train example:

1. full Qwen produces teacher logits under `no_grad`;
2. the native path produces student logits without executing a decoder block;
3. task cross-entropy and teacher KL train the Nolane cortex;
4. all Qwen parameters remain frozen and receive zero gradients.

The trained artifact is bound to the exact boundary-spec SHA-256.

## Quality court

```bash
python scripts/evaluate_native_boundary.py
```

The court compares untouched Qwen, L14 and L15 on the same frozen held-out protocol.

Promotion requires, among other gates:

- Qwen decoder forward calls = 0;
- Qwen decoder generation calls = 0;
- held-out improvement over untouched Qwen;
- non-inferiority to L14 within the registered budget;
- bounded Vietnamese/English anchor regression;
- prompt full-scan == token-incremental scan;
- deterministic native generation court;
- <=100K cortex parameters;
- unchanged, zero-gradient Qwen weights.

## Resource court

```bash
python scripts/benchmark_native_boundary_resources.py
```

The resource court measures native forward/generation against untouched Qwen and L14 and independently counts all Qwen decoder calls.

## Promotion

```bash
python scripts/decide_native_boundary_promotion.py \
  --quality runtime-data/l15-quality.json \
  --resources runtime-data/l15-resources.json
```

Quality and resource evidence must reference the identical checkpoint SHA and native-boundary spec SHA.

## Neural court evidence

CI instantiates real tiny `Qwen3ForCausalLM` models and proves:

- native forward produces logits while all Qwen decoder layers record exactly zero calls;
- native generation creates new tokens while all Qwen decoder layers record exactly zero calls;
- only Qwen embedding, final norm and LM head are invoked by native forward;
- full-prompt scan equals token-by-token recurrent scan;
- frozen full Qwen can teach the native cortex;
- training changes the Nolane cortex and no Qwen weights;
- Qwen gradient count remains zero;
- artifact roundtrip preserves cortex state and frozen boundary-spec lineage.

## Scientific boundary

L15 is **decoder-free**, not yet completely Qwen-free.

The model still reuses Qwen's token embedding matrix, final normalization and LM head. It also uses full Qwen as a teacher during training.

The next architectural boundary is to distill or replace these remaining input/output components so a standalone Nolane checkpoint can execute without loading Qwen model weights at inference.

Engineering closure does not imply production-quality parity with Qwen3-0.6B. Real promotion remains blocked until sufficient approved data and real held-out/resource evidence pass.
