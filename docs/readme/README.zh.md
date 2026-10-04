# Nolane AI Personal

**一个本地优先、能够跨时间保持连续性的个人 AI，而不是每次提示后重新开始的无状态聊天机器人。**

[English](../../README.md) · **中文**

## v1.0

- **Nolane Presence**：小型光球是抽象“面孔”，亮度和轻微运动来自真实情绪/活动状态。
- **记忆与未完话题**：查看 Nolane 记得什么、哪些对话仍未结束；记忆可以保留、编辑或忘记，并真正改变本地检索状态。
- **关系成长**：只显示“初识 → 熟悉 → 亲近”，由 closeness、trust、familiarity 推导，不使用等级或积分。
- **三步首次设置**：称呼、界面语言、对话风格。
- **主动对话**：Nolane 想主动说话时只显示一个小胶囊；打开才显示消息，忽略则保持安静。
- **轻微氛围**：背景仅随 mood 产生很轻的变化。
- **自定义身份外观**：可改名和头像，不改变模型 checkpoint 或运行时 identity。

## 本地优先隐私

状态、记忆和关系连续性以本地运行时为核心。Observable Mind 只展示受限的状态摘要，不展示原始隐藏推理。学习数据必须由用户明确审核后才能进入本地学习流程。

## 平台

Windows 使用 Tauri + 本地产品运行时；Android 使用 Tauri + 原生 Rust LocalMobile。Linux 提供运行时/研究工具；macOS 与 iOS 不属于 v1 产品目标。

## 开发

```bash
python -m pip install -e .
nolane-personal init
nolane-personal status
nolane-personal run --no-model --tick-seconds 5
```

产品客户端：

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

完整技术资料见 [Architecture](../ARCHITECTURE.md)、[Roadmap](../ROADMAP.md) 与 [v1 Living Presence](../V1-LIVING-PRESENCE.md)。

v1 软件发布必须通过同一 commit 的 CI-verified closure；它不会夸大为所有实体设备的电池、温度或 OEM 认证。

Apache-2.0.
