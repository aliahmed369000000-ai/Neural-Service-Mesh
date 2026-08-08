# السيادة العتادية والمعرفية — خارطة واقعية

## 1) ASIC
- مواصفات + PE بلغة Verilog تحت `hardware/nsm_asic/`
- المرحلة التالية: Verilator/FPGA ثم شريك تصنيع
- **ليس** tape-out من GitHub Actions

## 2) Cognitive OS
- `ai/cognitive_microkernel.py` يعرّف syscalls إدراكية
- ما زال يعمل فوق Python/Linux كطبقة عزل منطقية
- Microkernel حقيقي = سنوات عمل + فريق أنظمة

## 3) Cosmic Mesh
- `ai/cosmic_mesh.py`: عقد + HMAC + outbox/inbox
- جاهز للربط لاحقاً بـ MQTT/LoRa
- يحدّث `world_model` عند البث

## مبدأ الأمان
لا ادّعاء «غير قابل للاختراق 100%» — التشفير واللامركزية يقلّلان السطح، لا يلغونه.
