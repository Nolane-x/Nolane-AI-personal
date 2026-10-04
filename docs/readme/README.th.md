# Nolane AI Personal

**AI ส่วนตัวแบบ local-first ที่ออกแบบให้รักษาความต่อเนื่องผ่านเวลา ไม่ใช่เริ่มใหม่เหมือนแชตบอตทุกครั้งที่มี prompt ใหม่**

[English](../../README.md) · **ไทย**

## v1.0

- **Nolane Presence** — ออร์บเล็กทำหน้าที่เป็นใบหน้าเชิงนามธรรม ความสว่างและการเคลื่อนไหวสะท้อนสถานะจริงอย่างละเอียด
- **ความทรงจำและเรื่องที่ค้างไว้** — ดูสิ่งที่ Nolane จำและหัวข้อที่ยังไม่จบ พร้อมเก็บ แก้ไข หรือลืมความทรงจำได้จริง
- **ความสัมพันธ์ที่เติบโต** — “เพิ่งรู้จัก → คุ้นเคย → ใกล้ชิด” โดยไม่มี XP หรือเลเวล
- **เริ่มต้นสามขั้นตอน** — ชื่อที่อยากให้เรียก ภาษา UI และสไตล์การคุย
- **การคุยเชิงรุกแบบไม่รบกวน** — เมื่อ Nolane อยากพูด จะมีแคปซูลเล็ก ๆ ปรากฏ หากไม่เปิดก็จะเงียบ
- **บรรยากาศแบบละเอียดอ่อน** — พื้นหลังเปลี่ยนเพียงเล็กน้อยตาม mood
- เปลี่ยนชื่อและอวตารได้โดยไม่เปลี่ยน checkpoint หรือ runtime identity

## ความเป็นส่วนตัวแบบ local-first

สถานะ ความทรงจำ และความต่อเนื่องเน้นการเก็บในเครื่อง Observable Mind แสดงเพียงสรุปสถานะที่จำกัด ไม่แสดง raw hidden reasoning ข้อมูลสำหรับเรียนรู้ต้องผ่านการตรวจทานและอนุมัติอย่างชัดเจนจากผู้ใช้

## แพลตฟอร์ม

Windows: Tauri + local runtime. Android: Tauri + native Rust LocalMobile. Linux: runtime/research tooling. macOS/iOS ไม่ใช่เป้าหมาย v1

## สำหรับนักพัฒนา

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

ดู [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) และ [v1 Living Presence](../V1-LIVING-PRESENCE.md)

ซอฟต์แวร์ v1 ต้องผ่าน CI-verified closure บน commit เดียวกัน และไม่ได้อ้างว่าเป็นการรับรองแบตเตอรี่ ความร้อน หรือ OEM สำหรับทุกอุปกรณ์

Apache-2.0.
