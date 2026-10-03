from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


SAMPLER_SCHEMA = "NOLANE-V054-SEEDED-Q32-NUCLEUS-V1"
LOGIT_SCALE = 1_000
EXP_WEIGHT_SCALE = 1 << 40
PROBABILITY_SCALE = 1 << 32
U64_MASK = (1 << 64) - 1

_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB


def _round_half_up_positive(value: float) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("sampler quantization input must be finite and non-negative")
    return int(math.floor(value + 0.5))


def _round_half_away_from_zero(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("sampler logit must be finite")
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def splitmix64_next(state: int) -> tuple[int, int]:
    state = (int(state) + _SPLITMIX_GAMMA) & U64_MASK
    value = state
    value = ((value ^ (value >> 30)) * _SPLITMIX_MUL1) & U64_MASK
    value = ((value ^ (value >> 27)) * _SPLITMIX_MUL2) & U64_MASK
    value ^= value >> 31
    return state, value & U64_MASK


def _quantized_weights(
    logits: Sequence[float],
    *,
    temperature: float,
) -> list[int]:
    if not logits:
        raise ValueError("seeded sampler requires non-empty logits")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    values = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("logits contain non-finite value")

    quantized_logits = [
        _round_half_away_from_zero(value * LOGIT_SCALE)
        for value in values
    ]
    maximum = max(quantized_logits)
    exp_weights = [
        _round_half_up_positive(
            math.exp(
                ((value - maximum) / LOGIT_SCALE)
                / temperature
            )
            * EXP_WEIGHT_SCALE
        )
        for value in quantized_logits
    ]
    total_exp = sum(exp_weights)
    if total_exp <= 0:
        raise ValueError("softmax normalization failed")

    weights = [
        (
            value * PROBABILITY_SCALE
            + total_exp // 2
        )
        // total_exp
        for value in exp_weights
    ]
    if not any(weights):
        # The maximum logit has exp(0)=1, so this should only be reachable
        # for a vocabulary larger than the Q32 scale. Keep behavior defined.
        best = max(range(len(values)), key=lambda index: values[index])
        weights[best] = 1
    return weights


def _top_p_q32(top_p: float) -> int:
    value = float(top_p)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1]")
    return min(
        PROBABILITY_SCALE,
        _round_half_up_positive(value * PROBABILITY_SCALE),
    )


@dataclass(slots=True)
class SeededNucleusSampler:
    state: int
    temperature: float = 0.78
    top_p: float = 0.90

    def __post_init__(self) -> None:
        self.state = int(self.state) & U64_MASK
        # Validate policy eagerly.
        _ = _top_p_q32(self.top_p)
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature
        self.top_p = float(self.top_p)

    def sample(self, logits: Sequence[float]) -> int:
        weights = _quantized_weights(
            logits,
            temperature=self.temperature,
        )
        ranked = sorted(
            range(len(weights)),
            key=lambda token_id: (-weights[token_id], token_id),
        )
        total = sum(weights)
        if total <= 0:
            raise ValueError("quantized probability mass is empty")

        top_p_q32 = _top_p_q32(self.top_p)
        threshold = (total * top_p_q32) // PROBABILITY_SCALE

        retained: list[tuple[int, int]] = []
        cumulative_before = 0
        for token_id in ranked:
            weight = weights[token_id]
            if weight <= 0:
                continue
            if cumulative_before > threshold:
                break
            retained.append((token_id, weight))
            cumulative_before += weight

        if not retained:
            raise ValueError("top-p filter removed all probability mass")

        retained_total = sum(weight for _, weight in retained)
        self.state, random_value = splitmix64_next(self.state)
        draw = random_value % retained_total
        cumulative = 0
        for token_id, weight in retained:
            cumulative += weight
            if draw < cumulative:
                return token_id
        raise RuntimeError("seeded sampler draw escaped cumulative mass")

    def sample_tensor(self, logits) -> int:
        values = logits.detach().to("cpu", dtype=logits.new_empty(()).float().dtype)
        flattened = values.reshape(-1).tolist()
        return self.sample(flattened)


def sample_sequence(
    rows: Iterable[Sequence[float]],
    *,
    seed: int,
    temperature: float = 0.78,
    top_p: float = 0.90,
) -> tuple[list[int], int]:
    sampler = SeededNucleusSampler(
        state=seed,
        temperature=temperature,
        top_p=top_p,
    )
    tokens = [sampler.sample(row) for row in rows]
    return tokens, sampler.state
