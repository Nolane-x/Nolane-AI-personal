# Nolane AI Personal

**AI cá nhân local-first được thiết kế để sống xuyên thời gian, thay vì trở thành một chatbot mới sau mỗi prompt.**

[English](../../README.md) · **Tiếng Việt**

## Nolane khác chatbot thường ở đâu?

Nolane tách **khả năng sinh ngôn ngữ** khỏi **sự liên tục của bản thân AI**. Living Runtime giữ identity, thời gian, cảm xúc, mối quan hệ, ký ức, chuyện dang dở và khả năng chủ động. Language cortex chỉ là một cơ quan tạo ngôn ngữ có giới hạn.

### v1.0 có gì?

- **Nolane Presence** — orb nhỏ là “gương mặt trừu tượng”: vui/ấm sáng hơn, buồn/lo dịu xuống, tò mò chuyển động nhẹ, đang nghĩ thì pulse, nghỉ thì gần như đứng yên.
- **Memory & Threads** — xem Nolane đang nhớ gì và chuyện nào còn dang dở. Ký ức có thể **giữ, sửa hoặc quên** và thao tác này thay đổi bộ nhớ local thật.
- **Relationship growth** — chỉ hiện trạng thái tự nhiên **Mới quen → Quen thuộc → Gần gũi**, suy ra từ closeness/trust/familiarity; không XP, không level.
- **Onboarding 3 bước** — tên Nolane gọi bạn, ngôn ngữ UI và cách nói chuyện.
- **Proactive conversation** — khi Nolane tự muốn nói, một capsule nhỏ xuất hiện; mở thì xem, bỏ qua thì Nolane im. Không popup làm phiền.
- **Conversation atmosphere** — nền đổi cực nhẹ theo mood để tạo cảm giác hiện diện mà không biến thành theme màu mè.
- **Identity skin** — đổi tên và avatar AI mà không làm thay checkpoint, memory neural hay runtime identity.

## Quyền riêng tư local-first

Conversation state và memory được thiết kế để ở local. Người dùng có thể tắt memory. Observable Mind chỉ hiển thị tóm tắt state/hoạt động có giới hạn, **không hiển thị raw chain-of-thought**. Dữ liệu học chỉ được dùng khi người dùng duyệt rõ ràng qua workflow local review.

Khi người dùng quên một memory trên desktop, memory đó bị loại khỏi active SQLite memory store và không còn được cortex truy xuất. Audit của thao tác chỉ giữ digest/trạng thái cần thiết, không sao chép nội dung memory đã xóa.

## Nền tảng

| Nền tảng | Đường chạy v1 |
| --- | --- |
| Windows | Tauri desktop + local product runtime |
| Android | Tauri + native Rust LocalMobile |
| Linux | Runtime/research tooling, chưa phải packaged v1 target |
| macOS / iOS | Không nằm trong v1 |

## Chạy cho developer

```bash
python -m pip install -e .
nolane-personal init
nolane-personal status
```

Chạy state/memory/heartbeat mà không tải language model:

```bash
nolane-personal run --no-model --tick-seconds 5
```

Language dependencies:

```bash
python -m pip install -e '.[qwen]'
nolane-personal run
```

Product client nằm tại `apps/product-client/`.

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

## Model và kiến trúc

Dự án bắt đầu từ **Qwen3-0.6B** như upstream teacher/language source, sau đó nghiên cứu thay dần Transformer depth để tạo standalone Nolane inference path. Model weights không được commit vào GitHub.

Xem [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) và [v1 Living Presence](../V1-LIVING-PRESENCE.md).

## Release v1

Một bản software chỉ được coi là v1-ready khi cùng một commit vượt qua **CI-verified closure** cho Product Client, Living Runtime, Neural Shadow và Platform Crash. Closure này không tự động tuyên bố battery/thermal/OEM certification trên mọi điện thoại.

## Nguyên tắc

**UI nhỏ, hành vi sâu. State phải là state thật. Silence cũng là một hành động. Memory thuộc quyền kiểm soát của người dùng. Không giả vờ có ý thức sinh học. Release authority luôn fail-closed.**

Apache-2.0.
