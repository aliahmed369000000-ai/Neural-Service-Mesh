"""
تجهيز بيانات تدريب أغنى لغوياً من CKG — نسخة v3.
بدل القوالب الجامدة الخمسة السابقة (v1)، هنا 16 صياغة طبيعية متنوعة
لكل نوع علاقة، لتنويع لغوي حقيقي يساعد النموذج على تعميم أفضل بدل
حفظ نمط جملة واحد فقط.
"""
import json, random, re, pickle

random.seed(7)

with open('knowledge/cognitive_graph.json', encoding='utf-8') as f:
    ckg = json.load(f)

with open('knowledge_sources/quran/data/quran.json', encoding='utf-8') as f:
    _q = json.load(f)
surah_names = {str(s['number']): s['name'] for s in _q['surahs']['references']}

concepts = ckg['concepts']
relations = ckg['relations']


def clean_name(name):
    m = re.match(r'^سورة_(\d+)$', name)
    if m:
        return surah_names.get(m.group(1), name)
    m = re.match(r'^قرآن_(\d+):(\d+)$', name)
    if m:
        surah_num, ayah_num = m.group(1), m.group(2)
        surah_name = surah_names.get(surah_num, f"سورة_{surah_num}")
        return f"{surah_name} آية {ayah_num}"
    return name


# ── قوالب متنوعة لكل نوع علاقة (بدل قالب واحد جامد) ──────────────────────
REL_TEMPLATES = {
    'co_occurrence': [
        "{a} مرتبطة بـ {b} في القرآن الكريم",
        "يرد ذكر {a} مقترناً بـ {b} في آيات عديدة",
        "تتكرر {a} إلى جانب {b} في السياق القرآني",
        "من الملاحظ اقتران {a} بـ {b} في مواضع كثيرة",
    ],
    'root_link': [
        "{a} من نفس جذر {b} لغوياً",
        "يشترك {a} و{b} في الأصل الاشتقاقي",
        "الجذر اللغوي يجمع بين {a} و{b}",
    ],
    'narrative_sequence': [
        "{a} يأتي بعد {b} في تسلسل القصة",
        "تتوالى الأحداث من {b} إلى {a} في السرد القرآني",
        "{a} يلي {b} في الترتيب السردي",
    ],
    'semantic': [
        "{a} قريب المعنى من {b}",
        "يتقارب مفهوما {a} و{b} من حيث الدلالة",
        "{a} يحمل معنى شبيهاً بـ {b}",
        "من حيث المعنى، يتصل {a} بـ {b}",
    ],
    'thematic_cluster': [
        "{a} و{b} ينتميان لنفس المحور الموضوعي",
        "يجمع موضوع واحد بين {a} و{b}",
        "{a} يندرج مع {b} تحت نفس الفكرة العامة",
    ],
}
DEFAULT_TEMPLATES = ["{a} مرتبطة بـ {b}", "توجد علاقة بين {a} و{b}"]

CONCEPT_TEMPLATES = [
    "{name} مفهوم يندرج ضمن عنقود {cluster}",
    "يُصنَّف {name} من ضمن محور {cluster}",
    "{name} من المفاهيم المرتبطة بمجال {cluster}",
    "ينتمي {name} لموضوع {cluster} في القرآن الكريم",
]

sentences = []
for key, rel in relations.items():
    a, b = clean_name(rel.get('source', '')), clean_name(rel.get('target', ''))
    if not a or not b:
        continue
    rtype = rel.get('relation_type', 'co_occurrence')
    tmpl = random.choice(REL_TEMPLATES.get(rtype, DEFAULT_TEMPLATES))
    sentences.append(tmpl.format(a=a, b=b))

for name, meta in concepts.items():
    cluster = meta.get('cluster', 'معرفة')
    clean = clean_name(name)
    tmpl = random.choice(CONCEPT_TEMPLATES)
    sentences.append(tmpl.format(name=clean, cluster=cluster))

random.shuffle(sentences)
with open('ckg_sentences_v3.pkl', 'wb') as f:
    pickle.dump(sentences, f)

print(f"إجمالي الجمل (v3، صياغات متنوعة): {len(sentences)}")
print("أمثلة:")
for s in sentences[:8]:
    print(" -", s)
