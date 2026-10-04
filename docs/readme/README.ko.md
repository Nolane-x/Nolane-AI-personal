# Nolane AI Personal

**매 프롬프트마다 새로 시작하는 챗봇이 아니라, 시간의 흐름 속에서 정체성과 관계를 이어 가는 로컬 우선 개인 AI입니다.**

[English](../../README.md) · **한국어**

## v1.0

- **Nolane Presence** — 작은 오브가 추상적인 얼굴처럼 실제 mood/activity에 맞춰 밝기와 움직임을 바꿉니다.
- **기억과 이어갈 이야기** — Nolane의 로컬 기억과 끝나지 않은 주제를 보고, 기억을 보관·수정·삭제할 수 있습니다.
- **관계 성장** — “처음 → 익숙함 → 가까움”만 보여 주며 XP나 레벨은 없습니다.
- **3단계 온보딩** — 호칭, UI 언어, 대화 스타일만 설정합니다.
- **조용한 선제 대화** — Nolane이 먼저 말하고 싶을 때 작은 캡슐만 표시합니다. 열지 않으면 방해하지 않습니다.
- **대화 분위기** — mood에 따라 배경이 아주 미세하게 변합니다.
- 이름과 아바타 변경은 checkpoint나 runtime identity를 바꾸지 않습니다.

## 로컬 우선 개인정보 보호

상태와 기억은 로컬 중심이며, Observable Mind는 제한된 상태 요약만 보여 줍니다. 숨겨진 원시 추론은 노출하지 않습니다. 학습 데이터는 사용자가 명시적으로 검토한 경우에만 승인됩니다.

## 플랫폼

Windows는 Tauri + 로컬 runtime, Android는 Tauri + native Rust LocalMobile을 사용합니다. Linux는 runtime/research tooling을 제공하며 macOS/iOS는 v1 제품 대상이 아닙니다.

## 개발

```bash
python -m pip install -e .
nolane-personal init
nolane-personal run --no-model --tick-seconds 5
```

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

자세한 내용은 [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md), [v1 Living Presence](../V1-LIVING-PRESENCE.md)를 참고하세요.

v1 소프트웨어 릴리스는 동일 commit의 CI-verified closure가 필요하며 모든 실제 기기의 배터리·열·OEM 호환성을 자동으로 보증하지 않습니다.

Apache-2.0.
