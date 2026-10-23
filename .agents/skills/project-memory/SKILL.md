---
name: project-memory
description: "Use when starting a new session, recalling project context after context compaction, or updating project state and invariants before ending a task."
---

# Project-Memory Skill: Cross-Session Continuity & State Handoff

สกิลนี้ถูกออกแบบมาเพื่อแก้ไขปัญหา **Session Amnesia** (การลืมบริบทเมื่อเปิดแชทใหม่หรือเมื่อเกิด Context Compaction) โดยรวบรวมสถานะสำคัญ กฎเหล็ก และเป้าหมายปัจจุบันไว้ในไฟล์สรุปขนาดเล็กพิเศษ (`.agents/memory/ACTIVE_CONTEXT.md`)

---

## 1. Fast Context Recall (วิธีดึงความจำเมื่อเริ่มรอบใหม่)

เมื่อเปิดเซสชันใหม่ หรือหลังจากระบบแจ้งเตือนว่าเกิดการ Compaction:
1. **อ่านไฟล์หน่วยความจำหลักทันที:**
   เปิดดู [`ACTIVE_CONTEXT.md`](../../memory/ACTIVE_CONTEXT.md) ด้วย `view_file` (ไฟล์นี้มีความยาวไม่เกิน 60 บรรทัด หรือประมาณ ~400 tokens)
2. **คุณจะได้บริบทครบถ้วนทันทีโดยไม่ต้องอ่านไฟล์ย้อนหลังเป็นร้อยหน้า:**
   - ทราบกฎเหล็กและข้อห้าม (Invariants)
   - ทราบกลยุทธ์ที่กำลังทำงานอยู่ (CDCVIX15 บน 5 เหรียญหลัก)
   - ทราบสถานะการทดสอบล่าสุด (96/96 Unit Tests Pass)
   - ทราบสิ่งที่กำลังทำค้างอยู่และงานในขั้นตอนถัดไป

---

## 2. Context Checkpoint (วิธีอัปเดตความจำก่อนจบงาน)

เมื่อทำงานเสร็จในแต่ละช่วง หรือมีการเปลี่ยนแปลงทางสถาปัตยกรรมที่สำคัญ:
1. ปรับปรุงไฟล์ [`ACTIVE_CONTEXT.md`](../../memory/ACTIVE_CONTEXT.md)
2. โครงสร้างไฟล์ต้องคง 4 หมวดหลักเสมอ:
   - **`[1] Active Core Invariants`**: กฎคงสภาพที่ห้ามแก้ไขเด็ดขาด (เช่น Live Code Freeze, Whitelist, Isolated Margin)
   - **`[2] Current Milestone`**: งานล่าสุดที่เพิ่งทำเสร็จไป
   - **`[3] Verification Status`**: ผลการทดสอบล่าสุด (เช่น จำนวนข้อ unit test, commit hash)
   - **`[4] Immediate Next Steps`**: ขั้นตอนถัดไปที่ต้องทำต่อ

---

## 3. สคริปต์ช่วยอัปเดตอัตโนมัติ (Automated Helper)
สามารถเรียกใช้สคริปต์:
```bash
python .agents/skills/project-memory/scripts/update_memory.py --status
```
เพื่อตรวจสอบความสมบูรณ์ของหน่วยความจำประจำโปรเจกต์
