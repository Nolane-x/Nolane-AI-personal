from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path

from .engine import LivingEngine
from .qwen import QwenCortex
from .store import LivingStore


def _build_engine(args: argparse.Namespace) -> LivingEngine:
    store = LivingStore(Path(args.db))
    cortex = None
    observer = None
    if not getattr(args, "no_model", False):
        cortex = QwenCortex(args.model, device=args.device)
        if getattr(args, "social_observer", False):
            observer = cortex
    elif getattr(args, "social_observer", False):
        store.close()
        raise SystemExit("--social-observer requires the local model; remove --no-model")
    return LivingEngine(store, cortex=cortex, observer=observer)


def cmd_init(args: argparse.Namespace) -> int:
    store = LivingStore(args.db)
    engine = LivingEngine(store)
    print(f"identity_id={engine.state.identity_id}")
    print(f"state_version={engine.state.version}")
    print(f"db={Path(args.db).resolve()}")
    store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = LivingStore(args.db)
    state = store.load_state()
    if state is None:
        print("not initialized")
        store.close()
        return 1
    print(f"identity_id={state.identity_id}")
    print(f"version={state.version} tick={state.tick}")
    print(f"interactions={state.relationship.interaction_count}")
    print(f"open_threads={sum(1 for t in state.open_threads if t.unresolved)}")
    print(f"replay_transitions={len(store.replay_records())}")
    print(f"snapshot_digest={store.snapshot_digest()}")
    store.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    inbox: queue.Queue[str] = queue.Queue()

    def read_stdin() -> None:
        while True:
            try:
                line = input("you> ")
            except EOFError:
                inbox.put("/quit")
                return
            inbox.put(line)
            if line.strip() == "/quit":
                return

    threading.Thread(target=read_stdin, daemon=True).start()
    print(f"Nolane AI Personal alive: {engine.state.identity_id}")
    if args.social_observer:
        print("Social observer: Qwen proposal mode + deterministic validator")
    print("Commands: /quit, /status, /thread <topic>")

    try:
        while True:
            try:
                line = inbox.get(timeout=args.tick_seconds)
                command = line.strip()
                if command == "/quit":
                    break
                if command == "/status":
                    s = engine.state
                    print(f"state> v{s.version} tick={s.tick} social_drive={s.affect.social_drive:.2f} concern={s.affect.concern:.2f}")
                    continue
                if command.startswith("/thread "):
                    thread = engine.add_open_thread(command[len("/thread "):].strip())
                    print(f"state> opened {thread.thread_id}")
                    continue
                result = engine.handle_user_message(line)
                if result.observer_error:
                    print(f"state> observer rejected ({result.observer_error}); state preserved")
                if result.speech:
                    print(f"nolane> {result.speech}")
            except queue.Empty:
                result = engine.tick()
                if result.speech:
                    print(f"\nnolane> {result.speech}")
    except KeyboardInterrupt:
        pass
    finally:
        engine.store.close()
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nolane-personal")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize persistent living state")
    init.add_argument("--db", default="runtime-data/living.db")
    init.set_defaults(func=cmd_init)

    status = sub.add_parser("status", help="show persistent state identity")
    status.add_argument("--db", default="runtime-data/living.db")
    status.set_defaults(func=cmd_status)

    run = sub.add_parser("run", help="run the always-on local companion loop")
    run.add_argument("--db", default="runtime-data/living.db")
    run.add_argument("--model", default="models/Qwen3-0.6B")
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--tick-seconds", type=float, default=15.0)
    run.add_argument("--no-model", action="store_true", help="run state/memory/heartbeat without loading Qwen")
    run.add_argument("--social-observer", action="store_true", help="reuse Qwen for structured social proposals before validated persistence")
    run.set_defaults(func=cmd_run)
    return p


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
