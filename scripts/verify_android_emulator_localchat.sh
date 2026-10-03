#!/usr/bin/env bash
set -euo pipefail

APK_ROOT="${1:-artifacts/v058-emulator-apk}"
PACKAGE="ai.nolane.personal"
FIRST_MARKER="NOLANE_V058_FIRST_BOOT_PASS"
RESTART_MARKER="NOLANE_V058_RESTART_PASS"
FAIL_MARKER="NOLANE_V058_EMULATOR_COURT_FAIL"

APK="$(find "$APK_ROOT" -type f -name '*.apk' -print -quit)"
if [[ -z "$APK" ]]; then
  echo "v0.58 emulator APK not found under $APK_ROOT" >&2
  exit 1
fi

dump_failure_context() {
  echo "===== adb devices =====" >&2
  adb devices -l >&2 || true
  echo "===== process =====" >&2
  adb shell pidof "$PACKAGE" >&2 || true
  echo "===== recent logcat =====" >&2
  adb logcat -d -v time | tail -n 500 >&2 || true
}

wait_for_marker() {
  local marker="$1"
  local attempts="${2:-180}"
  local i
  for i in $(seq 1 "$attempts"); do
    local logs
    logs="$(adb logcat -d -v brief 2>/dev/null || true)"
    if grep -Fq "$FAIL_MARKER" <<<"$logs"; then
      echo "LocalMobile emulator probe reported failure" >&2
      grep -F "$FAIL_MARKER" <<<"$logs" >&2 || true
      dump_failure_context
      return 1
    fi
    if grep -Fq "$marker" <<<"$logs"; then
      grep -F "$marker" <<<"$logs"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for $marker" >&2
  dump_failure_context
  return 1
}

launch_app() {
  adb shell am force-stop "$PACKAGE" || true
  adb shell monkey     -p "$PACKAGE"     -c android.intent.category.LAUNCHER     1 >/dev/null
}

echo "Installing $APK"
adb install -r "$APK"

EXPECTED_VERSION="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(
    Path("apps/product-client/package.json").read_text(encoding="utf-8")
)
print(payload["version"])
PY
)"
VERSION="$(adb shell dumpsys package "$PACKAGE" \
  | tr -d '\r' \
  | sed -n 's/.*versionName=//p' \
  | head -n 1)"
if [[ "$VERSION" != "$EXPECTED_VERSION" ]]; then
  echo "Unexpected installed version: got=$VERSION expected=$EXPECTED_VERSION" >&2
  exit 1
fi

adb logcat -c
launch_app
wait_for_marker "$FIRST_MARKER"

FIRST_PID="$(adb shell pidof "$PACKAGE" | tr -d '\r' || true)"
if [[ -z "$FIRST_PID" ]]; then
  echo "App process disappeared after first-boot PASS" >&2
  dump_failure_context
  exit 1
fi

adb shell am force-stop "$PACKAGE"
sleep 2
if [[ -n "$(adb shell pidof "$PACKAGE" | tr -d '\r' || true)" ]]; then
  echo "App process did not stop cleanly" >&2
  dump_failure_context
  exit 1
fi

adb logcat -c
launch_app
wait_for_marker "$RESTART_MARKER"

SECOND_PID="$(adb shell pidof "$PACKAGE" | tr -d '\r' || true)"
if [[ -z "$SECOND_PID" ]]; then
  echo "App process disappeared after restart PASS" >&2
  dump_failure_context
  exit 1
fi

LOGS="$(adb logcat -d -v brief)"
if grep -Fq "$FAIL_MARKER" <<<"$LOGS"; then
  echo "Failure marker appeared after restart" >&2
  dump_failure_context
  exit 1
fi

echo "NOLANE_V058_ANDROID_EMULATOR_LOCALCHAT_COURT_PASS"
