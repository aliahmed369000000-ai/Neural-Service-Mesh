"""ملفات تعريف وكلاء NSM."""
import streamlit as st
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from ai.telemetry_store import TelemetryStore
from ui_components import render_kpi_cards, render_section_header

def _run_web_task(agent, task, url):
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"status": "blocked", "score": 0, "result": "الرابط يجب أن يكون صفحة عامة تبدأ بـ http أو https.", "sources": []}
    try:
        req = Request(url, headers={"User-Agent": "NSM-Agent/1.0"})
        with urlopen(req, timeout=12) as response:
            content = response.read(180000).decode("utf-8", errors="replace")
        class TextParser(HTMLParser):
            def __init__(self): super().__init__(); self.parts=[]
            def handle_data(self, data): self.parts.append(data)
        parser=TextParser(); parser.feed(content)
        text=" ".join(" ".join(parser.parts).split())[:5000]
        if not text: return {"status":"failed","score":10,"result":"لم يتم العثور على نص قابل للقراءة.","sources":[url]}
        result=f"المهمة: {task}\n\nتمت قراءة الصفحة العامة بنجاح.\n\nالمحتوى المستخرج:\n{text[:3500]}"
        score=min(100, 55 + (25 if len(text)>500 else 10) + (20 if task.lower() in text.lower() else 0))
        try:
            from ai.llm_fallback import LLMFallback
            prompt = f"المهمة المطلوبة: {task}\nالرابط المصدر: {url}\nمحتوى الصفحة:\n{text[:4200]}"
            generated = LLMFallback().generate(prompt, history=[], system_prompt="أنت وكيل بحث عربي. نفذ المهمة بدقة اعتماداً على محتوى الصفحة فقط. أجب بالعربية الفصحى، واذكر اسم الصفحة وثلاث نقاط رئيسية إن كان ذلك مطلوباً. لا تختلق معلومات ولا تستخدم Markdown معقداً.")
            summary = str(getattr(generated, "text", "") or "").strip()
            if summary and not getattr(generated, "error", None):
                result = summary[:5000]
                score = min(100, score + 20)
        except Exception:
            pass
        return {"status":"completed","score":score,"result":result,"sources":[url]}
    except Exception as exc:
        return {"status":"failed","score":0,"result":f"تعذر تنفيذ المهمة: {type(exc).__name__}","sources":[url]}

def render_agent_profiles():
    store = TelemetryStore(); profiles = store.list_agent_profiles(); settings = store.list_agent_settings()
    names = sorted({x['agent'] for x in profiles + settings}) or ['default']
    render_section_header('ملفات تعريف الوكلاء', 'هوية كل وكيل وقدراته وحالته التشغيلية', live=False)
    render_kpi_cards([
        {'label':'ملفات التعريف','value':len(profiles),'note':'محفوظة','accent':'var(--nsm-indigo)'},
        {'label':'مفعلة','value':sum(bool(x['enabled']) for x in profiles),'note':'جاهزة للعمل','accent':'var(--nsm-cyan)'},
        {'label':'متوقفة','value':sum(not bool(x['enabled']) for x in profiles),'note':'تحتاج مراجعة','accent':'var(--nsm-danger)'},
    ])
    selected = st.selectbox('اختر الوكيل', names, key='profile_agent')
    render_section_header("تكليف مهمة على الإنترنت", "اختبار بحث آمن على صفحة عامة مع حفظ الأداء")
    with st.form("web_task_form"):
        task = st.text_area("المهمة", placeholder="لخّص أهم النقاط في هذه الصفحة", max_chars=800)
        url = st.text_input("رابط الصفحة العامة", placeholder="https://example.com")
        if st.form_submit_button("تشغيل الاختبار", type="primary", use_container_width=True):
            if not task.strip() or not url.strip():
                st.error("أدخل المهمة والرابط أولاً.")
            else:
                with st.spinner("ينفذ الوكيل المهمة..."):
                    outcome = _run_web_task(selected, task, url)
                task_id = store.record_web_task(agent=selected, task=task, url=url, status=outcome["status"], score=outcome["score"], result=outcome["result"], sources=outcome["sources"])
                st.session_state["last_web_task_id"] = task_id
                st.session_state["last_web_task_agent"] = selected
                st.rerun()
    latest_tasks = store.list_web_tasks(agent=selected, limit=1)
    last_id = st.session_state.get("last_web_task_id")
    last_agent = st.session_state.get("last_web_task_agent")
    if latest_tasks and (last_agent == selected or last_id is None):
        saved = latest_tasks[0]
        st.success(f"اكتمل الاختبار بدرجة {saved['score']}/100" if saved["status"] == "completed" else "لم يكتمل الاختبار")
        st.text_area("نتيجة الوكيل", saved["result"], height=180, disabled=True, key=f"web_result_{saved['id']}")
        st.caption("المصدر: " + ", ".join(saved["sources"]))
    web_tasks = store.list_web_tasks(agent=selected, limit=20)
    if web_tasks:
        render_section_header("سجل اختبارات الإنترنت", "آخر المهام والدرجات")
        st.dataframe([{"المهمة": x["task"], "الحالة": x["status"], "الدرجة": x["score"], "الرابط": x["url"]} for x in web_tasks], use_container_width=True, hide_index=True)
    phone = store.get_agent_phone(selected)
    render_section_header("الهاتف التجريبي", "إعداد رقم Twilio العربي قبل التفعيل الفعلي")
    st.info("Twilio يحتاج حساباً ورصيداً تجريبياً ورقماً موثقاً. هذه الإعدادات تجهز الوكيل ولا تنشئ رقماً تلقائياً.")
    with st.form("agent_phone_form"):
        phone_number = st.text_input("رقم الهاتف", value=phone["phone_number"], placeholder="+1... أو رقم Twilio التجريبي")
        cols = st.columns(3)
        with cols[0]: provider = st.selectbox("المزود", ["twilio", "yemen_mobile"], index=0 if phone["provider"] == "twilio" else 1)
        with cols[1]: language = st.selectbox("لغة الرد", ["ar", "ar-SA", "ar-YE"], index=["ar", "ar-SA", "ar-YE"].index(phone["language"]))
        with cols[2]: phone_enabled = st.toggle("تفعيل الهاتف", value=bool(phone["enabled"]))
        webhook = st.text_input("مسار Webhook", value=phone["webhook_path"])
        if st.form_submit_button("حفظ إعدادات الهاتف", type="primary", use_container_width=True):
            store.save_agent_phone(agent=selected, phone_number=phone_number, provider=provider, language=language, webhook_path=webhook, enabled=phone_enabled)
            st.success("تم حفظ إعدادات الهاتف التجريبية."); st.rerun()
    calls = store.list_voice_calls(agent=selected, limit=50)
    render_section_header("سجل المكالمات", "آخر المكالمات الصوتية لهذا الوكيل")
    call_cols = st.columns(3)
    with call_cols[0]: st.metric("إجمالي المكالمات", len(calls))
    with call_cols[1]: st.metric("المكتملة", sum(x["status"] == "completed" for x in calls))
    with call_cols[2]: st.metric("متوسط المدة", f"{sum(x["duration_s"] for x in calls) / len(calls):.0f} ث" if calls else "0 ث")
    if calls:
        st.dataframe([{"الوقت": x["created_at"], "المتصل": x["caller"], "الحالة": x["status"], "المدة": f"{x["duration_s"]:.0f} ث", "النص": x["transcript"], "الرد": x["response"]} for x in calls], use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد مكالمات مسجلة بعد. سيظهر السجل عند ربط Webhook بمزود الاتصال.")

    current = store.get_agent_profile(selected)

    with st.form('agent_profile_form'):
        description = st.text_area('الوصف', value=current['description'], max_chars=500)
        capabilities_text = st.text_input('القدرات', value=', '.join(current['capabilities']), help='افصل القدرات بفواصل')
        enabled = st.toggle('الوكيل مفعّل', value=bool(current['enabled']))
        save = st.form_submit_button('حفظ ملف الوكيل', type='primary', use_container_width=True)
    if save:
        caps = [x.strip() for x in capabilities_text.split(',') if x.strip()]
        store.save_agent_profile(agent=selected, description=description, capabilities=caps, enabled=enabled)
        st.success('تم حفظ ملف الوكيل.'); st.rerun()
    if profiles:
        render_section_header('دليل الوكلاء', 'نظرة سريعة على الملفات والحالة')
        for profile in profiles:
            status = 'مفعّل' if profile['enabled'] else 'متوقف'
            st.markdown(f"**{profile['agent']}** · {status}\n\n{profile['description'] or 'لا يوجد وصف'}\n\nالقدرات: {', '.join(profile['capabilities']) or 'غير محددة'}")
            st.divider()
