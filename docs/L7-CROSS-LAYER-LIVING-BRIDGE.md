# L7 Cross-Layer Living Bridge

## Why L7 exists

L6 proved that a latent-conditioned neural adapter can receive real gradients through Qwen while all base-Qwen parameters remain frozen.

L7 asks a harder architectural question:

> Can a small recurrent non-Transformer pathway threaded through Qwen depth outperform the simpler L6 residual adapter under the same frozen personalization evidence?

The answer is not assumed. L7 has to earn promotion.

## Architecture

For selected decoder layers:

```text
persistent Living latent 32D
             |
             v
        latent projection
             |
             +-------------------+
                                 |
Qwen hidden -> hidden projection |
       |                         |
       +---- layer embedding ----+
                  |
                  v
             GRUCell 32D
                  |
          +-------+-------+
          |               |
          v               v
     residual vector   bounded gate
          |               |
          +-------*-------+
                  |
                  v
          Qwen hidden state
                  |
                  v
          next selected layer
```

The GRU state recurs across selected decoder layers during one model forward. It is reset from the persistent Living latent at the first selected layer of every forward, preventing accidental autograd carry-over between generation steps.

This adds a recurrent path *across Transformer depth*. It is not another prompt and not a fixed residual repeated at each layer.

## Default footprint

For hidden size 1024, latent 32D, bridge 32D and 64 possible layer identities:

**84,321 trainable parameters**

The production quality court caps L7 at **100,000 parameters**.

## Training boundary

- Qwen parameters: frozen, zero gradient
- Cross-Layer Living Bridge: trainable
- persistent 32D latent: input state, not a Qwen prompt
- personalization protocol: reused unchanged from L6
- trainer reads train split only

```bash
python scripts/train_living_bridge.py
```

## Evaluation

L7 is evaluated on the same frozen held-out personalization split and the same Vietnamese/English general anchor used by L6.

```bash
python scripts/evaluate_living_bridge.py
```

Promotion requires all of:

- held-out improvement over untouched Qwen;
- held-out improvement over the L6 Personal Cortex on the same examples;
- no unacceptable Vietnamese/English anchor regression;
- <=100K bridge parameters;
- base-Qwen weights unchanged;
- zero gradients on base-Qwen parameters.

The default required L7-vs-L6 held-out NLL improvement is **0.005**.

A recurrent bridge that is merely more complex than L6 is rejected.

## Generation

An unpromoted artifact can be exercised explicitly:

```bash
python scripts/generate_living_bridge.py \
  --prompt "Nay tôi hơi mệt."
```

It remains experimental until the real quality court passes.

## Engineering courts

CI uses a real tiny `Qwen3ForCausalLM` and verifies:

- exact analytical parameter count;
- recurrent trace traverses selected decoder layers;
- gates remain bounded;
- same prompt responds differently to different Living latents;
- bridge receives gradients;
- base Qwen receives zero gradients;
- bridge training loss decreases;
- held-out synthetic preference loss improves;
- generation works through the recurrent path;
- hooks are removed after use;
- artifact save/load preserves the bridge state digest;
- wrong base-model fingerprint or hidden size fails closed.

These courts prove the mechanism is real. They do not prove that L7 is better than L6 on real user data.
