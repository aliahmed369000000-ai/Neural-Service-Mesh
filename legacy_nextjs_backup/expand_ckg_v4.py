"""
expand_ckg_v4.py
================
المرحلة الثانية من توسعة الذاكرة الدلالية (CKG) — أقصى فهم ممكن.

يبني فوق CKG الحالي (173 مفهوم / 2149 علاقة) بإضافات تراكمية:

  1. مفاهيم جديدة (~80-100 مفهوم) تغطي ثغرات حقيقية:
     أنبياء/شخصيات إضافية، فقه، عقيدة، ظواهر طبيعية، مجتمع، تزكية.

  2. ربط الجذور العربية (633 جذراً) بالمفاهيم المناسبة
     → علاقات جديدة من نوع "root_link".

  3. ملامح السور (Surah Thematic Profiles):
     لكل سورة من 114 سورة، أبرز المفاهيم التي تظهر فيها
     → بُعد جديد "surah_profiles" في CKG.

  4. علاقات مجمّعية (thematic_cluster):
     مفاهيم تتشارك بقوة في نفس السور (لا فقط نفس الآية)
     → تكشف عن روابط موضوعية أوسع لم تُكتشف عبر التزامن في الآية.

  5. علاقات سردية (narrative_sequence):
     ربط شخصيات/قصص الأنبياء بمفاهيمها المرافقة (مثل: نوح ↔ هلاك الأمم).

كل الإضافات تراكمية فقط — لا حذف أو تعديل لأي مفهوم/علاقة موجودة من v3.
"""

import json, re, math, sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

_NOW = lambda: datetime.now(timezone.utc).isoformat()
_TASHKEEL = re.compile(r'[\u064B-\u065F\u0670\u0640\uFEFF]')
_ALEF     = re.compile(r'[أإآٱ]')
_TA       = re.compile(r'ة$')

def norm(t: str) -> str:
    t = _TASHKEEL.sub('', t)
    t = _ALEF.sub('ا', t)
    t = _TA.sub('ه', t)
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — مفاهيم جديدة (إضافة فقط، لا تكرار مع v3)
# ══════════════════════════════════════════════════════════════════════════
NEW_CONCEPTS = {
    # ── أنبياء وشخصيات إضافية ──
    "نبوة": {
        "هارون":      ["هرون"],
        "إسماعيل":    ["اسمعيل"],
        "إسحاق":      ["اسحق"],
        "يعقوب":      ["يعقوب"],
        "زكريا":      ["زكريا"],
        "يحيى":       ["يحيى"],
        "يونس":       ["يونس", "ذا النون"],
        "أيوب":       ["ايوب"],
        "شعيب":       ["شعيب"],
        "هود":        ["هود"],
        "صالح":       ["صلح"],
        "إدريس":      ["ادريس"],
        "ذو الكفل":   ["ذا الكفل"],
        "إلياس":      ["الياس"],
        "مريم":       ["مريم"],
        "آل عمران":   ["ال عمرن"],
        "أصحاب السبت": ["السبت"],
        "بلقيس":      ["ملكه سبا", "سبا"],
        "الخضر":      ["عبدا من عبادنا"],
    },

    # ── عقيدة وتوحيد إضافية ──
    "توحيد": {
        "الواحد الأحد": ["الاحد", "الواحد"],
        "الصمد":       ["الصمد"],
        "البصير":      ["البصير", "بصير"],
        "السميع":      ["السميع", "سميع"],
        "الغني":       ["الغني", "غني"],
        "التواب":      ["التواب", "توابا"],
        "الوهاب":      ["الوهاب"],
        "الفتاح":      ["فتاحا"],
        "النصير":      ["نصيرا", "النصير"],
        "الشاهد":      ["شهيدا", "الشهيد"],
        "المعين":      ["نستعين", "العون"],
        "القهار":      ["القهار", "قهار"],
        "الجبار":      ["الجبار"],
        "المتعالي":    ["المتعال"],
        "ذو الفضل":    ["ذو الفضل", "الفضل"],
        "ذو الجلال":   ["ذو الجلال"],
    },

    # ── فقه وأحكام عملية ──
    "فقه": {
        "وضوء وطهارة": ["تطهروا", "الطهور", "تيمموا"],
        "أكل وشرب":    ["كلوا", "اشربوا", "الطعام", "طعام"],
        "حلال وحرام":  ["حلالا", "حرام", "حرمت", "احل لكم"],
        "لحوم وذبائح": ["لحم الخنزير", "الميته", "ما ذكر اسم الله عليه"],
        "خمر ومسكرات": ["الخمر", "ميسر", "الانصاب"],
        "قصاص":        ["القصاص", "النفس بالنفس"],
        "شهادة":       ["اشهدوا", "الشهاده", "شهيدين"],
        "أيمان (حلف)": ["ايمانكم", "تحلفوا", "حلفوا"],
        "نذر":         ["النذر", "نذرت"],
        "كفارة":       ["كفاره", "تكفير"],
        "إحرام":       ["محرمون", "احرمتم"],
        "صيد":         ["الصيد", "صيدا"],
        "وصية":        ["الوصيه", "وصى"],
        "أيمان زواج":  ["مهر", "صداق", "اجورهن"],
    },

    # ── سلوك وأخلاق إضافية ──
    "أخلاق": {
        "أمانة المال":  ["الاموال بينكم بالباطل", "اكل اموال الناس"],
        "حسن الخلق":    ["خلق عظيم", "حسن الخلق"],
        "كظم الغيظ":    ["كاظمين الغيظ", "الغيظ"],
        "عطاء وبذل":    ["يبذلون", "اعطى"],
        "وقار":         ["السكينه", "وقارا"],
    },

    "سلوك": {
        "احتكار":      ["كنزوا", "الكنز"],
        "غلو":         ["لا تغلوا", "الغلو"],
        "تجسس":        ["لا تجسسوا"],
        "خداع":        ["يخدعون", "الخديعه"],
        "مكر":         ["المكر", "يمكرون"],
        "بغي":         ["البغي", "يبغون"],
        "طغيان":       ["الطغيان", "طغى"],
        "غدر":         ["الغدر"],
    },

    # ── آخرة إضافية ──
    "آخرة": {
        "ميزان الأعمال": ["الموازين", "وزن يومئذ"],
        "صراط الجحيم":   ["جسر", "صراط جهنم"],
        "خلود":          ["خالدين", "ابدا"],
        "حور عين":       ["حور عين", "كواعب"],
        "أنهار الجنة":   ["انهر من ماء", "انهار من لبن"],
        "زقوم وحميم":    ["الزقوم", "حميم"],
        "تطاير الصحف":   ["الكتاب يطير", "صحيفه"],
        "ندامة":         ["يا حسرتى", "الحسره"],
    },

    # ── كون وطبيعة إضافية ──
    "كون": {
        "زلازل":       ["الزلزله", "زلزال"],
        "بروق ورعد":   ["الرعد", "البرق", "الصواعق"],
        "ثلج وبرد":    ["البرد"],
        "عسل ولبن":    ["العسل", "اللبن"],
        "حديد ومعادن": ["الحديد", "الذهب", "الفضه"],
        "ملح وعذب":    ["ملح اجاج", "عذب فرات"],
        "ظل":          ["الظل", "ظلالها"],
        "فجر وشفق":    ["الفجر", "الشفق"],
        "مزن وسحاب":   ["السحاب", "المزن"],
    },

    # ── إيمان إضافية ──
    "إيمان": {
        "تسليم":        ["استسلموا", "اسلمت وجهي"],
        "ولاء":         ["تولوهم", "اولياء من دون الله"],
        "براءة":        ["براءه", "تبرأ"],
        "كبائر وصغائر": ["الكبائر", "اللمم"],
        "ردة":          ["يرتدد", "كفروا بعد ايمانهم"],
    },

    # ── مجتمع إضافية ──
    "مجتمع": {
        "رق وعبودية":  ["العبيد", "الرقاب", "ملك اليمين"],
        "سفر":         ["السفر", "ابن السبيل", "مسافرين"],
        "ضيافة":       ["الضيف", "اضافنا"],
        "أخوة":        ["اخوانكم", "الاخوه"],
        "جوار":        ["الجار", "الجوار"],
        "تجنيد ودفاع": ["النفير", "اعدوا لهم"],
        "غنائم":       ["الغنيمه", "الانفال"],
        "أسرى":        ["الاسارى", "الاساره"],
    },

    # ── معرفة إضافية ──
    "معرفة": {
        "وحي وإلهام":   ["نفث", "القاء في القلب"],
        "تأويل":        ["تأويله", "تأويلا"],
        "نسيان وذكر":   ["نسوا", "تنسى"],
        "أمثال القرآن": ["مثلا ما", "ضرب الله مثلا"],
        "آيات محكمة":   ["محكمات", "متشبهات"],
    },

    # ── قصص إضافية ──
    "قصص": {
        "أهل القرية":   ["القريه", "اصحب القريه"],
        "أصحاب الأخدود": ["الاخدود"],
        "أصحاب الرس":   ["الرس"],
        "إرم":          ["ارم ذات العماد"],
        "مدين":         ["مدين", "اهل مدين"],
        "سبأ":          ["سبا"],
        "أصحاب السفينة": ["الفلك", "السفينه"],
    },

    # ── روح إضافية ──
    "روح": {
        "وسوسة":       ["يوسوس", "الوسواس"],
        "غفلة":        ["غافلون", "الغفله"],
        "يقظة":        ["استيقظوا", "تذكروا فاذا"],
        "تفاؤل وأمل":  ["لا تيأسوا", "بشرى لكم"],
        "قنوط ويأس":   ["القنوط", "يؤوس"],
        "عزم":         ["العزم", "عزمت"],
    },

    # ── باطن إضافية ──
    "باطن": {
        "حضور القلب":  ["خشعت القلوب", "تطمئن القلوب"],
        "تفويض الأمر": ["وفوض امري الى الله", "اسلمت"],
        "محاسبة النفس": ["حاسبوا انفسكم"],
        "صدق النية":   ["مخلصين له الدين"],
    },
}


# ══════════════════════════════════════════════════════════════════════════
# تحميل البيانات الأساسية
# ══════════════════════════════════════════════════════════════════════════
print("Loading current CKG (v3)...")
ckg = json.load(open("knowledge/cognitive_graph.json", encoding="utf-8"))
print(f"  v3: {len(ckg['concepts'])} concepts, {len(ckg['relations'])} relations")

print("Loading Quran...")
data = json.load(open("knowledge_sources/quran/data/quran.json", encoding="utf-8"))
surahs_raw = data["surahs"]
if isinstance(surahs_raw, dict):
    surahs_raw = surahs_raw.get("references", list(surahs_raw.values()))

texts_norm, refs, surah_of = [], [], []
for s in surahs_raw:
    s_num = s.get("number", 0)
    for a in s.get("ayahs", []):
        txt = a.get("text", "").strip()
        if txt:
            texts_norm.append(norm(txt))
            refs.append(f"quran:{s_num}:{a.get('numberInSurah', 0)}")
            surah_of.append(s_num)

print(f"  Loaded {len(texts_norm)} ayahs across {len(surahs_raw)} surahs")

print("Loading Arabic roots index...")
roots_raw = json.load(open("knowledge/arabic_roots_index.json", encoding="utf-8"))
print(f"  {len(roots_raw)} roots")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — إضافة المفاهيم الجديدة وفحصها على القرآن
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Adding new concepts")
print("="*60)

NEW_CONCEPT_MAP = {}
for cluster, concepts in NEW_CONCEPTS.items():
    for cname, kws in concepts.items():
        if cname in ckg["concepts"]:
            continue  # تجنّب التكرار
        NEW_CONCEPT_MAP[cname] = {
            "cluster": cluster,
            "keywords": [norm(k) for k in kws],
        }

print(f"New candidate concepts: {len(NEW_CONCEPT_MAP)}")

new_concept_freq    = defaultdict(int)
new_concept_sources = defaultdict(set)

# نمسح القرآن مرة واحدة للمفاهيم الجديدة فقط (الحالية ثابتة من v3)
ayah_new_concepts = []
for i, txt in enumerate(texts_norm):
    found = set()
    for cname, info in NEW_CONCEPT_MAP.items():
        for kw in info["keywords"]:
            if kw in txt:
                found.add(cname)
                break
    ayah_new_concepts.append(found)
    for c in found:
        new_concept_freq[c] += 1
        new_concept_sources[c].add(refs[i])

# فلترة: الحد الأدنى 2 تكرارات
MIN_FREQ = 2
accepted_new = {c: f for c, f in new_concept_freq.items() if f >= MIN_FREQ}
print(f"Accepted new concepts (freq>=2): {len(accepted_new)}")
print(f"Rejected (freq<2): {len(new_concept_freq) - len(accepted_new)}")

# دمج في CKG
existing_max_freq = max((c.get("frequency", 1) for c in ckg["concepts"].values()), default=1)
for cname, freq in accepted_new.items():
    cluster = NEW_CONCEPT_MAP[cname]["cluster"]
    strength = round(math.log(freq + 1) / math.log(existing_max_freq + 1), 4)
    ckg["concepts"][cname] = {
        "name":       cname,
        "cluster":    cluster,
        "sources":    sorted(new_concept_sources[cname])[:30],
        "frequency":  freq,
        "strength":   strength,
        "first_seen": _NOW(),
        "last_seen":  _NOW(),
    }

print(f"Total concepts after Step 1: {len(ckg['concepts'])}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — علاقات تزامن للمفاهيم الجديدة (مع كل المفاهيم — قديمة وجديدة)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Co-occurrence relations for new concepts")
print("="*60)

# نحتاج لمسح كامل (مفاهيم قديمة + جديدة) لإيجاد التزامن مع الجديدة فقط
# لتفادي إعادة حساب علاقات v3 الحالية، نبني خريطة كلمات لكل مفهوم (قديم وجديد)
ALL_CONCEPT_KEYWORDS = {}
for cname in ckg["concepts"]:
    if cname in NEW_CONCEPT_MAP:
        ALL_CONCEPT_KEYWORDS[cname] = NEW_CONCEPT_MAP[cname]["keywords"]
    else:
        # نولّد كلمة مفتاحية أساسية من اسم المفهوم نفسه لمفاهيم v3
        ALL_CONCEPT_KEYWORDS[cname] = [norm(cname)]

accepted_new_set = set(accepted_new.keys())
co_occur_new = defaultdict(lambda: {"count": 0, "evidence": []})

for i, txt in enumerate(texts_norm):
    present = set()
    # المفاهيم الجديدة المقبولة الموجودة في هذه الآية (من الفحص السابق)
    for c in ayah_new_concepts[i]:
        if c in accepted_new_set:
            present.add(c)
    if not present:
        continue
    # تحقق من المفاهيم القديمة (v3) الموجودة في هذه الآية عبر sources
    # (أسرع: نتحقق من الكلمة المفتاحية البسيطة لاسم المفهوم القديم)
    for cname in ckg["concepts"]:
        if cname in accepted_new_set or cname in present:
            continue
        c_norm = norm(cname)
        if len(c_norm) >= 2 and c_norm in txt:
            present.add(cname)

    present_sorted = sorted(present)
    for a_idx in range(len(present_sorted)):
        for b_idx in range(a_idx + 1, len(present_sorted)):
            a, b = present_sorted[a_idx], present_sorted[b_idx]
            # نريد فقط أزواج تحتوي على الأقل مفهوماً جديداً واحداً
            if a not in accepted_new_set and b not in accepted_new_set:
                continue
            key = f"{a}||{b}"
            co_occur_new[key]["count"] += 1
            if len(co_occur_new[key]["evidence"]) < 5:
                co_occur_new[key]["evidence"].append(refs[i])

MIN_CO = 2
new_co_relations = 0
for key, info in co_occur_new.items():
    if info["count"] < MIN_CO:
        continue
    a, b = key.split("||")
    rel_key = f"{a}→{b}"
    if rel_key in ckg["relations"]:
        continue
    freq_a = ckg["concepts"].get(a, {}).get("frequency", 1)
    freq_b = ckg["concepts"].get(b, {}).get("frequency", 1)
    weight = round(min(info["count"] / max(min(freq_a, freq_b), 1), 1.0), 4)
    ckg["relations"][rel_key] = {
        "source":        a,
        "target":        b,
        "weight":        weight,
        "relation_type": "co_occurrence",
        "evidence":      info["evidence"],
        "count":         info["count"],
        "first_seen":    _NOW(),
        "last_seen":     _NOW(),
    }
    new_co_relations += 1

print(f"New co-occurrence relations: {new_co_relations}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — ربط الجذور العربية بالمفاهيم (root_link relations)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: Linking Arabic roots to concepts")
print("="*60)

root_link_count = 0
ROOTS_USED = set()

for root, info in roots_raw.items():
    if info.get("frequency", 0) < 5:
        continue
    top_token = norm(info.get("top_token", ""))
    tokens    = [norm(t) for t in info.get("tokens", [])]
    root_n    = norm(root)

    # نحاول إيجاد مفهوم يطابق هذا الجذر (مطابقة جزئية محافِظة)
    best_match = None
    for cname in ckg["concepts"]:
        c_norm = norm(cname)
        if len(c_norm) < 3:
            continue
        if c_norm == root_n or c_norm == top_token:
            best_match = cname
            break
        if c_norm in top_token or top_token in c_norm:
            best_match = cname
            break
        if any(c_norm in t or t in c_norm for t in tokens if len(t) >= 3):
            best_match = cname
            break

    if best_match:
        rel_key = f"root:{root}→{best_match}"
        if rel_key not in ckg["relations"]:
            ckg["relations"][rel_key] = {
                "source":        f"root:{root}",
                "target":        best_match,
                "weight":        round(min(info["frequency"] / 1000, 1.0), 4),
                "relation_type": "root_link",
                "evidence":      [],
                "count":         info["frequency"],
                "first_seen":    _NOW(),
                "last_seen":     _NOW(),
            }
            root_link_count += 1
            ROOTS_USED.add(root)

print(f"New root_link relations: {root_link_count}")
print(f"Distinct roots linked: {len(ROOTS_USED)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — ملامح السور (Surah Thematic Profiles)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: Building surah thematic profiles")
print("="*60)

# لكل مفهوم، نعرف في أي سور يظهر (من sources)
concept_surahs = defaultdict(Counter)
for cname, cdata in ckg["concepts"].items():
    for src in cdata.get("sources", []):
        try:
            surah_num = int(src.split(":")[1])
            concept_surahs[cname][surah_num] += 1
        except (IndexError, ValueError):
            continue

# نبني: لكل سورة، أبرز 5 مفاهيم
surah_profiles = defaultdict(Counter)
for cname, surah_counter in concept_surahs.items():
    for surah_num, count in surah_counter.items():
        surah_profiles[surah_num][cname] += count

surah_profiles_out = {}
for surah_num, counter in surah_profiles.items():
    top5 = counter.most_common(5)
    surah_profiles_out[str(surah_num)] = [
        {"concept": c, "weight": w} for c, w in top5
    ]

ckg["surah_profiles"] = surah_profiles_out
print(f"Surah profiles built for {len(surah_profiles_out)} surahs")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — علاقات مجمّعية (thematic_cluster) عبر تشارك السور
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: Thematic cluster relations (shared-surah co-occurrence)")
print("="*60)

# لكل سورة، المفاهيم الموجودة فيها (من concept_surahs)
surah_concepts = defaultdict(set)
for cname, surah_counter in concept_surahs.items():
    for surah_num in surah_counter:
        surah_concepts[surah_num].add(cname)

pair_surah_count = defaultdict(int)
for surah_num, concepts_in_surah in surah_concepts.items():
    concepts_sorted = sorted(concepts_in_surah)
    for i in range(len(concepts_sorted)):
        for j in range(i + 1, len(concepts_sorted)):
            a, b = concepts_sorted[i], concepts_sorted[j]
            pair_surah_count[(a, b)] += 1

# فقط أزواج تتشارك في 8+ سور (إشارة موضوعية قوية) ولم تُربط بعد
MIN_SHARED_SURAHS = 10
thematic_count = 0
total_surahs = 114
for (a, b), shared in pair_surah_count.items():
    if shared < MIN_SHARED_SURAHS:
        continue
    rel_key = f"{a}→{b} (thematic)"
    if rel_key in ckg["relations"]:
        continue
    weight = round(shared / total_surahs, 4)
    ckg["relations"][rel_key] = {
        "source":        a,
        "target":        b,
        "weight":        weight,
        "relation_type": "thematic_cluster",
        "evidence":      [],
        "count":         shared,
        "first_seen":    _NOW(),
        "last_seen":     _NOW(),
    }
    thematic_count += 1

print(f"New thematic_cluster relations: {thematic_count}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — علاقات سردية (narrative_sequence) لشخصيات وقصص الأنبياء
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6: Narrative sequence relations (prophets ↔ themes)")
print("="*60)

NARRATIVE_LINKS = [
    ("نوح",     "هلاك الامم"),
    ("نوح",     "اصحاب السفينه"),
    ("لوط",     "هلاك الامم"),
    ("عاد",     "هلاك الامم"),
    ("ثمود",    "هلاك الامم"),
    ("فرعون",   "هلاك الامم"),
    ("موسى",    "فرعون"),
    ("موسى",    "هارون"),
    ("يوسف",    "اخوه"),
    ("ابرهيم",  "اسماعيل"),
    ("ابرهيم",  "اسحق"),
    ("اسحق",    "يعقوب"),
    ("يعقوب",   "يوسف"),
    ("زكريا",   "يحيى"),
    ("مريم",    "عيسى"),
    ("عيسى",    "ال عمرن"),
    ("يونس",    "صبر"),
    ("ايوب",    "صبر"),
    ("سليمن",   "بلقيس"),
    ("سليمن",   "حكمه"),
    ("داود",    "حكمه"),
    ("شعيب",    "مدين"),
    ("صلح",     "ثمود"),
    ("هود",     "عاد"),
    ("ذو القرنين", "ياجوج ماجوج"),
    ("اصحاب الكهف", "ايمان"),
    ("ادم",     "خلق"),
    ("ادم",     "توبه"),
]

narrative_count = 0
for a, b in NARRATIVE_LINKS:
    # تأكد من وجود المفهومين في CKG (بالاسم المطبّع المطابق)
    a_match = next((c for c in ckg["concepts"] if norm(c) == norm(a)), None)
    b_match = next((c for c in ckg["concepts"] if norm(c) == norm(b)), None)
    if not a_match or not b_match:
        continue
    rel_key = f"{a_match}→{b_match} (narrative)"
    if rel_key in ckg["relations"]:
        continue
    ckg["relations"][rel_key] = {
        "source":        a_match,
        "target":        b_match,
        "weight":        0.6,
        "relation_type": "narrative_sequence",
        "evidence":      [],
        "count":         0,
        "first_seen":    _NOW(),
        "last_seen":     _NOW(),
    }
    narrative_count += 1

print(f"New narrative_sequence relations: {narrative_count}")


# ══════════════════════════════════════════════════════════════════════════
# حفظ CKG النهائي
# ══════════════════════════════════════════════════════════════════════════
ckg.setdefault("meta", {})
ckg["meta"]["version"] = "4.0-deep-expansion"
ckg["meta"]["built_at"] = _NOW()
ckg["meta"]["total_concepts"] = len(ckg["concepts"])
ckg["meta"]["total_relations"] = len(ckg["relations"])
ckg["meta"]["relation_types"] = sorted({r.get("relation_type", "") for r in ckg["relations"].values()})
ckg["meta"]["roots_linked"] = len(ROOTS_USED)
ckg["meta"]["surah_profiles_count"] = len(surah_profiles_out)

out_path = Path("knowledge/cognitive_graph.json")
out_path.write_text(json.dumps(ckg, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
size = out_path.stat().st_size

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"  Concepts          : {len(ckg['concepts'])}")
print(f"  Relations         : {len(ckg['relations'])}")
print(f"  Relation types    : {ckg['meta']['relation_types']}")
print(f"  Roots linked      : {len(ROOTS_USED)}")
print(f"  Surah profiles    : {len(surah_profiles_out)}")
print(f"  File size         : {size:,} bytes ({size/1024:.1f} KB)")

# Top concepts by frequency
print("\nTop 20 concepts by frequency:")
top20 = sorted(ckg["concepts"].items(), key=lambda x: -x[1].get("frequency", 0))[:20]
for rank, (name, cdata) in enumerate(top20, 1):
    print(f"  {rank:2d}. {name:<20} freq={cdata.get('frequency', 0):4d}  cluster={cdata.get('cluster','')}")
