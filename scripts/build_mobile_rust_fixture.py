from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.factorized_boundary import (
    FactorizedBoundaryConfig,
    FactorizedBoundaryModule,
)
from nolane_personal.mobile_factorized import (
    build_mobile_factorized_modules,
    export_mobile_factorized_package,
)
from nolane_personal.mobile_prompt_contract import (
    freeze_product_prompt_contract,
    render_product_prompt,
    sha256_file,
    write_product_prompt_contract,
)
from nolane_personal.mobile_persistent_state import (
    build_persistent_mobile_state,
    write_persistent_mobile_state,
)
from nolane_personal.product_profile import ProductProfile
from nolane_personal.state import LivingState, OpenThread
from nolane_personal.seeded_sampling import (
    SAMPLER_SCHEMA,
    SeededNucleusSampler,
)
from nolane_personal.standalone_model import StandaloneNolaneLM


SOURCE_SHA = "7" * 64


class FixtureChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return (
            str(messages[0]["content"])
            + " "
            + str(messages[1]["content"])
        )


def fixture_model() -> StandaloneNolaneLM:
    torch.manual_seed(503)
    boundary = FactorizedBoundaryModule(
        FactorizedBoundaryConfig(
            vocab_size=29,
            hidden_size=10,
            rank=4,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    cortex = DeepRecurrentStateSpaceCortex(
        10,
        DeepRecurrentCortexConfig(
            latent_dim=4,
            state_dim=3,
            virtual_steps=3,
            max_virtual_steps=4,
            max_abs_gate=0.2,
            initial_gate=0.04,
            slow_decay_floor=0.55,
        ),
        seed=509,
    )
    return StandaloneNolaneLM(
        boundary,
        cortex,
        [0.25, -0.15, 0.05, 0.35],
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    package = root / "package"
    root.mkdir(parents=True, exist_ok=True)

    model = fixture_model()
    init_state, token_step, _contract = build_mobile_factorized_modules(model)
    export_mobile_factorized_package(
        model,
        package,
        source_checkpoint_sha256=SOURCE_SHA,
    )

    vocab = {f"tok{index}": index for index in range(29)}
    tokenizer = Tokenizer(
        WordLevel(
            vocab=vocab,
            unk_token="tok0",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer_config_path = root / "tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps({"chat_template": "fixture-system-space-user"}) + "\n",
        encoding="utf-8",
    )

    prompt_contract = freeze_product_prompt_contract(
        FixtureChatTokenizer(),
        tokenizer_json=tokenizer_path,
        tokenizer_config_json=tokenizer_config_path,
    )
    prompt_contract_path = write_product_prompt_contract(
        prompt_contract,
        root / "prompt-contract.json",
    )
    system_prompt = "tok1 tok6"
    user_prompt = "tok11 tok4 tok17"

    prompt = "tok1 tok6 tok11 tok4 tok17"
    assert render_product_prompt(
        prompt_contract,
        system_prompt,
        user_prompt,
    ) == prompt
    tokens = tokenizer.encode(prompt).ids
    assert tokens == [1, 6, 11, 4, 17]

    latent = torch.tensor([model.latent], dtype=torch.float32)
    rows = []
    generated = []
    stopped_on_eos = False
    with torch.no_grad():
        state = init_state(latent)
        initial_state = state[0].tolist()
        logits = None
        for token in tokens:
            logits, state = token_step(
                torch.tensor([token], dtype=torch.long),
                state,
                latent,
            )
            rows.append(
                {
                    "token_id": token,
                    "logits": logits[0].tolist(),
                    "state": state[0].tolist(),
                }
            )

        assert logits is not None
        for _ in range(4):
            token = int(torch.argmax(logits[0]).item())
            generated.append(token)
            if token == 2:
                stopped_on_eos = True
                break
            logits, state = token_step(
                torch.tensor([token], dtype=torch.long),
                state,
                latent,
            )

    sampling_seed = 0xC0FFEE1234567890
    sampling_temperature = 0.78
    sampling_top_p = 0.90
    sampled_max_new_tokens = 8
    sampled = []
    sampled_stopped_on_eos = False

    with torch.no_grad():
        sampled_state = init_state(latent)
        sampled_logits = None
        for token in tokens:
            sampled_logits, sampled_state = token_step(
                torch.tensor([token], dtype=torch.long),
                sampled_state,
                latent,
            )
        assert sampled_logits is not None
        sampler = SeededNucleusSampler(
            state=sampling_seed,
            temperature=sampling_temperature,
            top_p=sampling_top_p,
        )
        for _ in range(sampled_max_new_tokens):
            token = sampler.sample(
                sampled_logits[0].detach().float().cpu().tolist()
            )
            sampled.append(token)
            if token == 2:
                sampled_stopped_on_eos = True
                break
            sampled_logits, sampled_state = token_step(
                torch.tensor([token], dtype=torch.long),
                sampled_state,
                latent,
            )

        standalone_seeded = model.generate(
            input_ids=torch.tensor([tokens], dtype=torch.long),
            max_new_tokens=sampled_max_new_tokens,
            do_sample=True,
            temperature=sampling_temperature,
            top_p=sampling_top_p,
            sampling_seed=sampling_seed,
            eos_token_id=2,
            pad_token_id=0,
        )
        standalone_new = standalone_seeded[
            0,
            len(tokens):,
        ].tolist()
        if standalone_new != sampled:
            raise RuntimeError(
                "Standalone seeded generation drifted from mobile token-step "
                f"sampler: standalone={standalone_new} mobile={sampled}"
            )

    mobile_profile = ProductProfile(
        preferred_name="Thuận",
        language="vi",
        response_length="compact",
        conversation_style="natural",
        personal_instruction="Nói ngắn gọn và nhớ dự án.",
    )
    mobile_state = LivingState(identity_id="identity-v055-mobile")
    mobile_state.relationship.closeness = 0.71
    mobile_state.relationship.trust = 0.82
    mobile_state.relationship.familiarity = 0.63
    mobile_state.relationship.interaction_count = 17
    mobile_state.affect.valence = 0.2
    mobile_state.affect.energy = 0.4
    mobile_state.affect.playfulness = 0.1
    mobile_state.affect.concern = 0.05
    mobile_state.affect.irritation = 0.0
    mobile_state.open_threads = [
        OpenThread(
            thread_id="v055",
            topic="Hoàn thành persistent mobile state bridge",
        )
    ]
    persistent_payload = build_persistent_mobile_state(
        source_checkpoint_sha256=SOURCE_SHA,
        latent=latent[0].tolist(),
        profile=mobile_profile,
        state=mobile_state,
        memories=[
            "Người dùng đang xây Nolane.",
            "Tiếp tục hoàn thành v0.55 trước LocalMobile.",
        ],
    )
    write_persistent_mobile_state(
        root / "persistent-state.json",
        persistent_payload,
        expected_source_checkpoint_sha256=SOURCE_SHA,
        expected_latent_dim=4,
    )
    write_persistent_mobile_state(
        root / "bootstrap-state.json",
        persistent_payload,
        expected_source_checkpoint_sha256=SOURCE_SHA,
        expected_latent_dim=4,
    )
    (root / "localmobile-manifest.json").write_text(
        json.dumps(
            {
                "schema": "NOLANE-V056-LOCALMOBILE-BUNDLE-V1",
                "source_checkpoint_sha256": SOURCE_SHA,
                "prompt_contract_file_sha256": sha256_file(
                    prompt_contract_path
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "schema": "NOLANE-V050-MOBILE-RUST-GOLDEN-V1",
        "source_checkpoint_sha256": SOURCE_SHA,
        "latent": latent[0].tolist(),
        "tokens": tokens,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_contract_file_sha256": sha256_file(prompt_contract_path),
        "prompt_token_ids": tokens,
        "max_new_tokens": 4,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "stopped_on_eos": stopped_on_eos,
        "generation_final_state": state[0].tolist(),
        "initial_state": initial_state,
        "steps": rows,
        "atol": 2e-5,
        "rtol": 2e-5,
        "seeded_sampler_schema": SAMPLER_SCHEMA,
        "sampling_seed": sampling_seed,
        "sampling_temperature": sampling_temperature,
        "sampling_top_p": sampling_top_p,
        "sampled_max_new_tokens": sampled_max_new_tokens,
        "sampled_generated_token_ids": sampled,
        "sampled_generated_text": tokenizer.decode(sampled),
        "sampled_stopped_on_eos": sampled_stopped_on_eos,
        "sampled_final_state": sampled_state[0].tolist(),
    }
    (root / "fixture.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
