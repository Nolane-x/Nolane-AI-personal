from copy import deepcopy
import hashlib

import pytest

from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import (
    build_personalization_protocol,
    examples_for_split,
    verify_personalization_protocol,
)


def _examples(n=10):
    return [
        PersonalizationExample(
            prompt=f"prompt-{i}",
            target=f"target-{i}",
            latent=[0.01 * i] * 32,
            language="vi" if i % 2 == 0 else "en",
        )
        for i in range(n)
    ]


def test_personalization_protocol_is_chronological_and_frozen():
    examples = _examples(10)
    protocol = build_personalization_protocol(examples, dataset_sha256="dataset")
    assert protocol["counts"] == {"total": 10, "train": 7, "dev": 1, "test": 2}
    assert [row["index"] for row in protocol["splits"]["train"]] == list(range(7))
    assert [row["index"] for row in protocol["splits"]["test"]] == [8, 9]
    verify_personalization_protocol(protocol, dataset_sha256="dataset")
    assert [x.prompt for x in examples_for_split(examples, protocol, "test")] == ["prompt-8", "prompt-9"]


def test_personalization_protocol_rejects_dataset_and_row_drift():
    protocol = build_personalization_protocol(_examples(6), dataset_sha256="dataset-a")
    with pytest.raises(ValueError, match="dataset digest mismatch"):
        verify_personalization_protocol(protocol, dataset_sha256="dataset-b")

    tampered = deepcopy(protocol)
    tampered["splits"]["test"][0]["target"] = "changed"
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        verify_personalization_protocol(tampered, dataset_sha256="dataset-a")


def test_personalization_protocol_group_split_never_crosses_source_group():
    examples=_examples(8)
    groups=[
        hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in ("a","a","b","c","d","e","f","g")
    ]
    protocol=build_personalization_protocol(
        examples,
        dataset_sha256="dataset-grouped",
        source_group_sha256=groups,
    )
    assert protocol["split_strategy"]=="source_group_chronological_v1"
    split_groups={
        split:{row["source_group_sha256"] for row in protocol["splits"][split]}
        for split in ("train","dev","test")
    }
    assert not (split_groups["train"] & split_groups["dev"])
    assert not (split_groups["train"] & split_groups["test"])
    assert not (split_groups["dev"] & split_groups["test"])
    assert protocol["counts"]["test"]>=1
    verify_personalization_protocol(protocol,dataset_sha256="dataset-grouped")


def test_personalization_protocol_rejects_too_few_source_groups():
    examples=_examples(7)
    one=hashlib.sha256(b"same-conversation").hexdigest()
    with pytest.raises(ValueError,match="at least 3 distinct source groups"):
        build_personalization_protocol(
            examples,
            dataset_sha256="dataset-grouped",
            source_group_sha256=[one]*7,
        )
