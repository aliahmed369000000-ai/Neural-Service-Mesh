# NSM ASIC — خارطة تصميم الرقاقة (ليست تصنيعاً)

## النطاق الواقعي
هذا المجلد يوثّق **مواصفات RTL أولية** لنوى الحساب في `NeuralCore` / التوجيه 784→…→4.
**لا يُنتج رقاقة سيليكون جاهزة** من هذا المستودع — التصنيع يحتاج شريك Foundry وFlow كامل (synthesis, P&R, tape-out).

## النوى المرشّحة للتسريع
| النواة | العملية | ملاحظة |
|--------|---------|--------|
| MATMUL_784 | ضرب مصفوفة 784×N | أعلى كثافة حساب |
| RELU_BANK | تفعيل متوازي | رخيص |
| SOFTMAX4 | 4 مخرجات توجيه | صغير جداً |
| CKG_HASH | مطابقة مفاهيم محلية | ذاكرة ROM مدمجة اختيارية |

## تدفق مستقبلي
1. محاكاة RTL (`iverilog` / Verilator)
2. FPGA prototype (Xilinx/Lattice)
3. ASIC tape-out مع شريك

## تشغيل المحاكاة (إن وُجد iverilog)
```bash
cd hardware/nsm_asic
iverilog -o sim_matmul matmul_pe.v tb_matmul_pe.v && vvp sim_matmul
```
