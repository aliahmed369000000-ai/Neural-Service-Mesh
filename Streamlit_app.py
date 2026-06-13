import streamlit as st
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(page_title="Neural Service Mesh", layout="wide", page_icon="🧠")

st.markdown("""
<style>
body { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Neural Service Mesh")
st.markdown("### لوحة المراقبة المعرفية")

# Load CKG stats
try:
    with open("knowledge/cognitive_graph.json", encoding="utf-8") as f:
        ckg = json.load(f)
    concepts = len(ckg.get("concepts", {}))
    relations = len(ckg.get("relations", {}))
except:
    concepts = 0
    relations = 0

# Load training stats
try:
    with open("checkpoints/deep_network_training_summary.json", encoding="utf-8") as f:
        training = json.load(f)
    train_steps = training.get("train_steps", 0)
    last_loss = training.get("last_loss", 0)
except:
    train_steps = 0
    last_loss = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("📚 مفاهيم CKG", concepts)
col2.metric("🔗 علاقات CKG", relations)
col3.metric("🔄 خطوات التدريب", train_steps)
col4.metric("📉 آخر خسارة", f"{last_loss:.4f}" if last_loss else "—")

st.divider()

st.subheader("🔎 استعلام عن مفهوم")
query = st.text_input("ابحث في CKG", placeholder="مثال: الله، القرآن...")
if st.button("بحث") and query:
    try:
        concepts_dict = ckg.get("concepts", {})
        if query in concepts_dict:
            st.success(f"✅ وُجد: {query}")
            st.json(concepts_dict[query])
        else:
            st.warning(f"❌ غير موجود في CKG: {query}")
    except Exception as e:
        st.error(str(e))

st.divider()
st.caption("Neural Service Mesh v18 — لوحة المراقبة المعرفية")
