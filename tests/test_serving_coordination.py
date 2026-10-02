from pathlib import Path

import pytest

from nolane_personal.serving_coordination import (
    ServingCoordinator,
    ServingLeasePolicy,
    ServingSession,
)


def pointer(generation: int, label: str, previous=None):
    checkpoint = (label * 64)[:64]
    pointer_sha = ((label.upper() + "0") * 64)[:64]
    return {
        "schema": "NOLANE-L33-ACTIVE-CHECKPOINT-POINTER-V1",
        "authority": "TRANSACTIONAL_CHECKPOINT_SUBSTRATE_NO_AUTONOMOUS_PROD_AUTHORITY",
        "generation": generation,
        "checkpoint_sha256": checkpoint,
        "artifact_relpath": f"artifacts/{checkpoint}",
        "previous_pointer_sha256": previous,
        "transaction_id": f"tx-{generation}",
        "transition": "INITIALIZE" if generation == 0 else "UPDATE",
        "rollback_target_generation": None,
        "created_at": "2026-10-02T00:00:00+00:00",
        "pointer_sha256": pointer_sha,
    }


class FakeRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._pointers = {
            0: pointer(0, "a"),
            1: pointer(1, "b"),
            2: pointer(2, "c"),
        }
        self._active = 0
        for p in self._pointers.values():
            (root / p["artifact_relpath"]).mkdir(parents=True, exist_ok=True)

    def active_pointer(self):
        return dict(self._pointers[self._active])

    def pointer_for_generation(self, generation):
        if int(generation) not in self._pointers:
            raise ValueError("unknown generation")
        return dict(self._pointers[int(generation)])

    def artifact_path_for_pointer(self, p):
        expected = self._pointers[int(p["generation"])]
        if p != expected:
            raise ValueError("pointer mismatch")
        return self.root / expected["artifact_relpath"]

    def activate(self, generation):
        self._active = int(generation)


def loader(bundle, p):
    return {"bundle": str(bundle), "generation": p["generation"]}, p["checkpoint_sha256"]


def bad_loader(bundle, p):
    return object(), "f" * 64


def coordinator(tmp_path, *, min_live=2, lease_seconds=30.0):
    registry = FakeRegistry(tmp_path / "registry")
    coord = ServingCoordinator(
        registry,
        root=tmp_path / "serving",
        policy=ServingLeasePolicy(
            lease_seconds=lease_seconds,
            min_live_processes=min_live,
        ),
    )
    return registry, coord


def test_two_process_reload_never_grants_serve_during_split_brain(tmp_path):
    registry, coord = coordinator(tmp_path)
    p1 = ServingSession(coord, process_id="worker-1", loader=loader)
    p2 = ServingSession(coord, process_id="worker-2", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    p1.start(now=now)
    p2.start(now=now)

    assert coord.assess_convergence(now=now)["status"] == "PASS"
    assert p1.gate(now=now)["status"] == "SERVE"
    assert p2.gate(now=now)["status"] == "SERVE"

    registry.activate(1)
    assert p1.gate(now=now)["status"] == "DRAIN_RELOAD_REQUIRED"
    assert p2.gate(now=now)["status"] == "DRAIN_RELOAD_REQUIRED"

    assert p1.poll_reload(now=now) is True
    split = coord.assess_convergence(now=now)
    assert split["status"] == "BLOCKED"
    assert "split_brain_detected" in split["reasons"]
    assert "live_processes_not_on_active_pointer" in split["reasons"]
    assert p1.gate(now=now)["status"] == "WAITING_FOR_PEERS"
    assert p2.gate(now=now)["status"] == "DRAIN_RELOAD_REQUIRED"

    assert p2.poll_reload(now=now) is True
    assert coord.assess_convergence(now=now)["status"] == "PASS"
    assert p1.gate(now=now)["status"] == "SERVE"
    assert p2.gate(now=now)["status"] == "SERVE"


def test_reload_ack_fails_if_active_changes_during_model_load(tmp_path):
    registry, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    session.start(now=now)
    registry.activate(1)
    plan = coord.reload_plan("worker")
    assert plan["target_generation"] == 1

    registry.activate(2)
    with pytest.raises(RuntimeError, match="active checkpoint changed"):
        coord.ack_reload(
            "worker",
            plan=plan,
            loaded_checkpoint_sha256=plan["target_checkpoint_sha256"],
            now=now,
        )

    lease = coord.load_lease("worker")
    assert lease["loaded_generation"] == 0
    assert coord.serving_gate("worker", now=now)["status"] == "DRAIN_RELOAD_REQUIRED"


def test_expired_old_worker_does_not_block_live_converged_worker(tmp_path):
    registry, coord = coordinator(
        tmp_path,
        min_live=1,
        lease_seconds=10.0,
    )
    old = ServingSession(coord, process_id="old-worker", loader=loader)
    live = ServingSession(coord, process_id="live-worker", loader=loader)
    t0 = "2026-10-02T00:00:00+00:00"
    old.start(now=t0)
    live.start(now=t0)

    registry.activate(1)
    live.poll_reload(now="2026-10-02T00:00:05+00:00")
    live.poll_reload(now="2026-10-02T00:00:12+00:00")

    receipt = coord.assess_convergence(
        now="2026-10-02T00:00:12+00:00"
    )
    assert receipt["status"] == "PASS"
    assert receipt["live_processes"] == 1
    assert receipt["expired_processes"] == 1
    assert live.gate(now="2026-10-02T00:00:12+00:00")["status"] == "SERVE"
    assert old.gate(now="2026-10-02T00:00:12+00:00")["status"] == "LEASE_EXPIRED"


def test_minimum_live_process_gate_blocks_single_survivor(tmp_path):
    registry, coord = coordinator(
        tmp_path,
        min_live=2,
        lease_seconds=10.0,
    )
    one = ServingSession(coord, process_id="one", loader=loader)
    two = ServingSession(coord, process_id="two", loader=loader)
    one.start(now="2026-10-02T00:00:00+00:00")
    two.start(now="2026-10-02T00:00:00+00:00")
    one.poll_reload(now="2026-10-02T00:00:12+00:00")

    receipt = coord.assess_convergence(
        now="2026-10-02T00:00:12+00:00"
    )
    assert receipt["status"] == "BLOCKED"
    assert "insufficient_live_serving_processes" in receipt["reasons"]
    assert one.gate(now="2026-10-02T00:00:12+00:00")["status"] == "WAITING_FOR_PEERS"


def test_wrong_checkpoint_loader_never_registers_or_acks(tmp_path):
    _, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=bad_loader)
    with pytest.raises(ValueError, match="wrong checkpoint"):
        session.start(now="2026-10-02T00:00:00+00:00")
    with pytest.raises(ValueError, match="not registered"):
        coord.load_lease("worker")


def test_lease_tamper_is_detected(tmp_path):
    _, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=loader)
    session.start(now="2026-10-02T00:00:00+00:00")
    path = coord._lease_path("worker")
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace('"loaded_generation":0', '"loaded_generation":9'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lease digest mismatch"):
        coord.load_lease("worker")


def test_reload_plan_is_bound_to_process(tmp_path):
    registry, coord = coordinator(tmp_path, min_live=1)
    first = ServingSession(coord, process_id="first", loader=loader)
    second = ServingSession(coord, process_id="second", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    first.start(now=now)
    second.start(now=now)
    registry.activate(1)
    plan = coord.reload_plan("first")

    with pytest.raises(ValueError, match="process binding mismatch"):
        coord.ack_reload(
            "second",
            plan=plan,
            loaded_checkpoint_sha256=plan["target_checkpoint_sha256"],
            now=now,
        )


def test_process_ids_are_not_exposed_in_convergence_receipt(tmp_path):
    _, coord = coordinator(tmp_path, min_live=1)
    secret_id = "host-very-private-process-id"
    session = ServingSession(coord, process_id=secret_id, loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    session.start(now=now)
    receipt = coord.assess_convergence(now=now)
    assert secret_id not in str(receipt)
    assert receipt["privacy"]["contains_raw_process_id"] is False


def test_request_model_is_blocked_until_cluster_converges(tmp_path):
    registry, coord = coordinator(tmp_path)
    p1 = ServingSession(coord, process_id="worker-1", loader=loader)
    p2 = ServingSession(coord, process_id="worker-2", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    p1.start(now=now)
    p2.start(now=now)
    assert p1.model_for_request(now=now)["generation"] == 0

    registry.activate(1)
    with pytest.raises(RuntimeError, match="DRAIN_RELOAD_REQUIRED"):
        p1.model_for_request(now=now)

    p1.poll_reload(now=now)
    with pytest.raises(RuntimeError, match="WAITING_FOR_PEERS"):
        p1.model_for_request(now=now)

    p2.poll_reload(now=now)
    assert p1.model_for_request(now=now)["generation"] == 1
    assert p2.model_for_request(now=now)["generation"] == 1


def test_convergence_blocks_if_active_pointer_changes_mid_assessment(
    tmp_path,
    monkeypatch,
):
    registry, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    session.start(now=now)

    original = registry.active_pointer
    calls = {"count": 0}

    def flipping_active():
        calls["count"] += 1
        if calls["count"] == 2:
            registry.activate(1)
        return original()

    monkeypatch.setattr(registry, "active_pointer", flipping_active)
    receipt = coord.assess_convergence(now=now)
    assert receipt["status"] == "BLOCKED"
    assert "active_pointer_changed_during_convergence" in receipt["reasons"]


def test_request_fence_rejects_pointer_change_after_gate_pass(tmp_path):
    registry, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    session.start(now=now)
    gate = session.gate(now=now)
    assert gate["status"] == "SERVE"

    registry.activate(1)
    with pytest.raises(RuntimeError, match="fence invalidated"):
        coord.verify_request_fence("worker", gate)


def test_model_for_request_closes_post_gate_pointer_race(
    tmp_path,
    monkeypatch,
):
    registry, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    session.start(now=now)
    original_gate = session.gate

    def gate_then_flip(*, now=None):
        gate = original_gate(now=now)
        assert gate["status"] == "SERVE"
        registry.activate(1)
        return gate

    monkeypatch.setattr(session, "gate", gate_then_flip)
    with pytest.raises(RuntimeError, match="fence invalidated"):
        session.model_for_request(now=now)


def test_request_context_allows_drain_but_blocks_next_request(tmp_path):
    registry, coord = coordinator(tmp_path, min_live=1)
    session = ServingSession(coord, process_id="worker", loader=loader)
    now = "2026-10-02T00:00:00+00:00"
    session.start(now=now)

    with session.request_model(now=now) as model:
        assert model["generation"] == 0
        registry.activate(1)
        assert model["generation"] == 0

    with pytest.raises(RuntimeError, match="DRAIN_RELOAD_REQUIRED"):
        with session.request_model(now=now):
            pass

    assert session.poll_reload(now=now) is True
    with session.request_model(now=now) as model:
        assert model["generation"] == 1
