from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "android_root",
        help="Path to generated src-tauri/gen/android directory",
    )
    args = parser.parse_args()

    root = Path(args.android_root)
    matches = list(
        root.glob(
            "buildSrc/src/main/java/**/kotlin/BuildTask.kt"
        )
    )
    if len(matches) != 1:
        raise SystemExit(
            "expected exactly one generated Tauri BuildTask.kt; "
            f"found {len(matches)}"
        )

    path = matches[0]
    text = path.read_text(encoding="utf-8")

    old_executable = 'val executable = """node""";'
    new_executable = 'val executable = """npm""";'
    old_args = (
        'val args = listOf("tauri", "android", '
        '"android-studio-script");'
    )
    new_args = (
        'val args = listOf("run", "--", "tauri", "android", '
        '"android-studio-script");'
    )

    if old_executable not in text:
        raise SystemExit(
            "generated Tauri BuildTask no longer uses the expected node "
            "launcher; review upstream template before changing this court"
        )
    if old_args not in text:
        raise SystemExit(
            "generated Tauri BuildTask arguments changed; review upstream "
            "template before changing this court"
        )

    patched = text.replace(
        old_executable,
        new_executable,
        1,
    ).replace(
        old_args,
        new_args,
        1,
    )
    if patched == text:
        raise SystemExit("Tauri Android BuildTask patch made no change")

    path.write_text(patched, encoding="utf-8")
    verify = path.read_text(encoding="utf-8")
    if new_executable not in verify or new_args not in verify:
        raise SystemExit("Tauri Android BuildTask patch verification failed")

    print(f"patched Tauri Android CLI bridge: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
