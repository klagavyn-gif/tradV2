---
name: token-saver
description: "Use when inspecting large files, debugging logs/datasets, refactoring code, or whenever you need to optimize token consumption and prevent context compaction."
---

# Token-Saver Skill: Ultra-Lean Context & Token Optimization

สกิลนี้ถูกออกแบบมาเพื่อควบคุมการใช้ Context Window และลด Token Consumption อย่างมีนัยสำคัญในโปรเจกต์ขนาดใหญ่ (เช่น tradV2 ที่มี `trad.py` กว่า 11,000 บรรทัด และไฟล์ข้อมูล JSON ขนาดใหญ่)

---

## 1. กฎเหล็ก 4 ข้อในการประหยัด Token (The 4 Golden Rules)

### กฎข้อที่ 1: "Locate Before Reading" (ค้นหาจุดก่อนอ่านเสมอ)
- **ห้าม** ใช้ `view_file` เปิดดูทั้งไฟล์กับไฟล์ที่มีขนาดเกิน 150 บรรทัด (เช่น `trad.py`, `config.py`, `AGENTS.md`)
- ให้ใช้ `grep_search` ค้นหาชื่อฟังก์ชันหรือคีย์เวิร์ดที่ต้องการก่อน เพื่อให้ทราบหมายเลขบรรทัด (`LineNumber`)
- เมื่อจะอ่าน ให้ใช้ `view_file` โดยระบุ `StartLine` และ `EndLine` เสมอ (อ่านทีละ 30–70 บรรทัดเฉพาะจุดที่เกี่ยวข้อง)

### กฎข้อที่ 2: "Compute, Don't Read" (คำนวณผ่านสคริปต์แทนการอ่านข้อมูลดิบ)
- เมื่อต้องวิเคราะห์ข้อมูลสถิติ ประวัติผลการเทรด หรือ Log ขนาดใหญ่ (เช่น `.data/realized_outcomes.json`, `risk_state.json`) **ห้ามอ่านไฟล์ดิบเข้า context เด็ดขาด** เพราะจะทำให้เปลือง token หลายหมื่น token
- ให้เขียนสคริปต์ Python สั้นๆ (5-10 บรรทัด) ในโฟลเดอร์ `scratch/` เพื่อคำนวณและสั่งพิมพ์ออกมาเฉพาะข้อสรุป (Summary Metrics) ที่ต้องการไม่เกิน 15 บรรทัด

### กฎข้อที่ 3: "Surgical Diffs" (แก้ไขเฉพาะจุด ไม่เขียนทับทั้งไฟล์)
- ใช้ `replace_file_content` หรือ `multi_replace_file_content` โดยระบุบล็อกโค้ดที่ต้องการเปลี่ยนให้สั้นกระชับที่สุด
- หลีกเลี่ยงการใช้ `write_to_file` แบบเขียนทับทั้งไฟล์เดิม เว้นแต่เป็นการสร้างไฟล์ใหม่

### กฎข้อที่ 4: "Subagent Delegation for Bulk Tasks" (แยกงานหนักไปให้ Subagent)
- สำหรับงานที่ต้องสำรวจไฟล์จำนวนมาก การสแกนทั้ง repository หรือการค้นคว้าเว็บ ให้เรียกใช้ Subagent เพื่อให้กระบวนการสืบค้นดิบถูกเก็บไว้ใน Subagent Session และส่งกลับมายังบทสนทนาหลักเฉพาะข้อสรุป (Final Synthesis)

---

## 2. ขั้นตอนปฏิบัติเมื่อต้องทำงานกับไฟล์ใหญ่ใน tradV2

| งานที่ต้องทำ | วิธีที่เปลือง Token (ห้ามทำ ❌) | วิธีที่ประหยัด Token สูงสุด (ต้องทำ ✅) |
| :--- | :--- | :--- |
| **ดูฟังก์ชันใน `trad.py`** | เรียก `view_file` ที่บรรทัด 1 ถึง 800 | ใช้ `grep_search` หา `def function_name` แล้ว `view_file` เฉพาะช่วง 40 บรรทัดของฟังก์ชันนั้น |
| **ตรวจประวัติ Realized Trades** | เปิดอ่าน `realized_outcomes.json` ทั้งไฟล์ | รัน `python -c "import json; ... print(summary)"` เพื่อดึงเฉพาะ Win Rate และ Total PnL |
| **ค้นหากฎใน `AGENTS.md`** | เลื่อนอ่านตั้งแต่บรรทัด 1 | ใช้ `grep_search` หาหัวข้อที่ต้องการ เช่น `## 42.` แล้วเปิดดูเฉพาะส่วนนั้น |
| **รันและตรวจ Unit Tests** | ปริ้นต์ log ทั้งหมดที่ไม่จำเป็น | รัน `python -m unittest discover -s tests -p "test_*.py"` และดูเฉพาะสรุปบรรทัดสุดท้าย |

---

## 3. รูปแบบการตอบสนองแบบ High-Density (Concise Communication)
- ตัดคำเกริ่นนำ คำขอบคุณ และคำขอโทษที่ไม่จำเป็นออก
- สรุปคำตอบเป็น Bullet Points, Code Links และตารางสรุป
- ชี้เป้าไปยังไฟล์หรือ Artifact แทนการคัดลอกโค้ดขนาดยาวซ้ำในข้อความแชท
