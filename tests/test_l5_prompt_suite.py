import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_l5_shadow_prompt_suite_is_frozen_unique_and_nonempty():
    path = ROOT / "research" / "L5-SHADOW-PROMPTS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "NOLANE-L5-SHADOW-PROMPTS-V1"
    prompts = payload["prompts"]
    assert len(prompts) >= 10
    assert len(prompts) == len(set(prompts))
    assert all(isinstance(item, str) and item.strip() for item in prompts)
    assert any(any(ord(ch) > 127 for ch in item) for item in prompts)
    assert any(item.isascii() for item in prompts)
