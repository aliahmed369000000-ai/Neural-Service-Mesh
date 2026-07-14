# NSM MCP Server

يعرض هذا السيرفر بعض أدوات مشروع Neural Service Mesh (NSM) كأدوات
[MCP](https://modelcontextprotocol.io) قياسية، بحيث يقدر أي عميل MCP
(Claude Desktop، Claude Code، أو أي IDE يدعم MCP) يستخدمها مباشرة
بدون المرور عبر واجهة Streamlit.

## الأدوات المتاحة

| الأداة | الوصف |
|---|---|
| `quran_lookup(surah, ayah)` | جلب نص آية بعينها عبر رقم السورة ورقم الآية |
| `quran_search(query, limit)` | بحث نصّي عن آيات تحتوي كلمة/عبارة معيّنة |
| `classify_harm(text)` | تصنيف نص حسب نطاق الأذى المحتمل (مبني على `ai/harm_classifier.py`) |

## التشغيل محلياً

```bash
pip install -r requirements.txt
python mcp_server/server.py
```

السيرفر يستخدم transport من نوع `stdio`، وهو الافتراضي لمعظم عملاء MCP
المحلية (Claude Desktop، Claude Code).

## الإضافة إلى Claude Desktop

أضف المقطع التالي إلى `claude_desktop_config.json` (مع تعديل المسار):

```json
{
  "mcpServers": {
    "nsm": {
      "command": "python",
      "args": ["/path/to/Neural-Service-Mesh/mcp_server/server.py"]
    }
  }
}
```

## ملاحظات

- الأدوات الحالية للقراءة فقط (read-only) — لا تعدّل أي بيانات في `knowledge/`.
- بيانات القرآن تُقرأ مباشرة من ملفات `knowledge/quran_chunk_*.json` و
  `knowledge/quran_index.json` الموجودة بالفعل في المشروع، بدون أي نسخ إضافي.
- هذا أول إصدار (v1) — يمكن التوسّع لاحقاً بإضافة أدوات أخرى من `ai/`
  (مثل مصنّف الأذى ثنائي اللغة أو أدوات المصادر المعرفية) بنفس النمط.
