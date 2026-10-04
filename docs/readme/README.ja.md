# Nolane AI Personal

**毎回のプロンプトでリセットされるのではなく、時間をまたいで連続性を保つローカル優先のパーソナル AI。**

[English](../../README.md) · **日本語**

## v1.0

- **Nolane Presence** — 小さなオーブが抽象的な顔として、実際の mood/activity に応じて静かに変化します。
- **記憶と未完の話** — Nolane が覚えていることと続きのある話を確認し、記憶を残す・編集する・忘れさせることができます。
- **関係の成長** — 「出会ったばかり → 慣れてきた → 親しい」。XP やレベルはありません。
- **3ステップの初回設定** — 呼び名、UI 言語、話し方だけを設定します。
- **主动ではなく自然なプロアクティブ会話** — 発言したい時は小さなカプセルだけを表示し、無視すれば静かなままです。
- **会話の空気感** — 背景は mood に合わせてごくわずかに変化します。
- 名前とアバターは変更できますが、モデル checkpoint や runtime identity は変えません。

## ローカル優先

状態、記憶、関係性はローカル中心です。Observable Mind は安全な状態要約のみを表示し、生の hidden reasoning は表示しません。学習用データはユーザーが明示的にレビューしたものだけを扱います。

## 対応プラットフォーム

Windows: Tauri + local runtime。Android: Tauri + native Rust LocalMobile。Linux は runtime/research tooling。macOS/iOS は v1 対象外です。

## 開発

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

詳細は [Architecture](../ARCHITECTURE.md)、[Roadmap](../ROADMAP.md)、[v1 Living Presence](../V1-LIVING-PRESENCE.md)。

v1 のソフトウェアリリースは同一 commit の CI-verified closure を必要とし、全端末のバッテリー・熱・OEM 動作まで証明したとは主張しません。

Apache-2.0.
