import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.preprocessing import preprocess_for_sentiment

# ============ KONFIGURASI ============
MODEL_DIR = "zacheldy/transjakarta-indobert-sentiment"
MAX_LENGTH = 128
ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}
DATASET_PATH = "data/dataset_05_indobert_predicted_from_roberta_label.csv"
TOPIC_PATH = "data/bertopic_topic_summary_all_sentiments_roberta_max10.csv"

COLOR_MAP = {"negatif": "#e74c3c", "netral": "#95a5a6", "positif": "#27ae60"}
EMOJI_MAP = {"negatif": "😠", "netral": "😐", "positif": "😊"}

CONTOH_TEKS = [
    "Busnya lama banget, udah nunggu 30 menit di halte gak dateng-dateng",
    "Halte deket rumah lumayan bersih, tapi kadang suka penuh pas jam pulang kerja",
    "Makasih TransJakarta udah nambah armada, sekarang jadi lebih cepet nungguinnya",
]

st.set_page_config(page_title="Analisis Sentimen TransJakarta", page_icon="🚌", layout="wide")

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    .result-card h2 { margin: 0; font-size: 2rem; }
    .result-card p { margin: 0.3rem 0 0 0; opacity: 0.9; }
    .prob-label { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_data
def load_dashboard_data():
    df_pred = pd.read_csv(DATASET_PATH)
    df_topic = pd.read_csv(TOPIC_PATH)
    return df_pred, df_topic


tokenizer, model = load_model()
df_pred, df_topic = load_dashboard_data()

if "history" not in st.session_state:
    st.session_state.history = []


def predict_sentiment(text_raw: str):
    text_clean = preprocess_for_sentiment(text_raw)
    inputs = tokenizer(text_clean, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], float(probs[pred_id]), text_clean, probs


# ============ SIDEBAR ============
with st.sidebar:
    st.header("ℹ️ Tentang Riset Ini")
    st.markdown("""
    **Skripsi:** Analisis Sentimen dan Pemodelan Topik terhadap Layanan TransJakarta
    di Twitter/X Menggunakan IndoBERT dan BERTopic

    **Dataset:** 14.114 tweet (setelah cleaning)

    **Model:** IndoBERT (fine-tuned)
    """)
    st.divider()
    st.subheader("📊 Performa Model")
    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy", "86.40%")
    col_b.metric("Weighted F1", "86.36%")
    st.metric("Macro F1", "78.17%")
    st.caption("Evaluasi pada test set (10% dari total data, stratified split)")


# ============ HEADER ============
st.title("🚌 Analisis Sentimen & Topik Layanan TransJakarta")
st.caption("Coba sendiri model IndoBERT, atau jelajahi hasil analisis dari 14.114 tweet publik")

tab1, tab2, tab3 = st.tabs(["🔍 Coba Sendiri", "📊 Dashboard Hasil Riset", "📖 Tentang Metodologi"])

# ---------- TAB 1: LIVE INFERENCE ----------
with tab1:
    st.subheader("Coba Prediksi Sentimen")

    st.markdown("**Coba contoh cepat:**")
    cols = st.columns(len(CONTOH_TEKS))
    example_clicked = None
    for i, contoh in enumerate(CONTOH_TEKS):
        if cols[i].button(f"Contoh {i+1}", use_container_width=True):
            example_clicked = contoh

    user_input = st.text_area(
        "Atau masukkan teks/tweet sendiri:",
        value=example_clicked if example_clicked else "",
        placeholder="Contoh: busnya lama banget, udah nunggu 30 menit di halte",
        height=100,
    )

    if st.button("🔍 Prediksi Sentimen", type="primary", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            label, confidence, text_clean, probs = predict_sentiment(user_input)
            color = COLOR_MAP[label]

            st.markdown(
                f"""
                <div class="result-card" style="background-color:{color};">
                    <h2>{EMOJI_MAP[label]} {label.upper()}</h2>
                    <p>Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Distribusi Probabilitas:**")
            for lbl in ["negatif", "netral", "positif"]:
                idx = {"negatif": 0, "netral": 1, "positif": 2}[lbl]
                st.markdown(f"<div class='prob-label'>{EMOJI_MAP[lbl]} {lbl.capitalize()} — {probs[idx]*100:.1f}%</div>", unsafe_allow_html=True)
                st.progress(float(probs[idx]))

            with st.expander("🔧 Detail teknis (hasil preprocessing)"):
                st.code(text_clean)

            st.session_state.history.insert(0, {
                "teks": user_input[:60] + ("..." if len(user_input) > 60 else ""),
                "sentimen": label,
                "confidence": f"{confidence*100:.1f}%"
            })
            st.session_state.history = st.session_state.history[:5]

    if st.session_state.history:
        st.divider()
        st.markdown("**🕒 Riwayat Prediksi Terakhir**")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📁 Prediksi Massal (Upload CSV)")
    st.caption("Upload file CSV dengan satu kolom bernama `text` untuk prediksi banyak tweet sekaligus")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "text" not in df_upload.columns:
                st.error("File CSV harus punya kolom bernama 'text'.")
            else:
                with st.spinner("Memproses prediksi..."):
                    results = [predict_sentiment(t)[:2] for t in df_upload["text"].astype(str)]
                    df_upload["sentimen_prediksi"] = [r[0] for r in results]
                    df_upload["confidence"] = [f"{r[1]*100:.1f}%" for r in results]

                st.success(f"Selesai! {len(df_upload)} teks berhasil diprediksi.")
                st.dataframe(df_upload, use_container_width=True, hide_index=True)

                csv_result = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hasil Prediksi",
                    data=csv_result,
                    file_name="hasil_prediksi_sentimen.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# ---------- TAB 2: STATIC DASHBOARD ----------
with tab2:
    st.subheader("Distribusi Sentimen (14.114 tweet)")
    sentiment_counts = df_pred["sentiment_label"].value_counts().reindex(
        ["negatif", "netral", "positif"]
    ).fillna(0)
    st.bar_chart(sentiment_counts)

    col1, col2, col3 = st.columns(3)
    total = len(df_pred)
    col1.metric("😠 Negatif", f"{sentiment_counts['negatif']:.0f}", f"{sentiment_counts['negatif']/total*100:.1f}%")
    col2.metric("😐 Netral", f"{sentiment_counts['netral']:.0f}", f"{sentiment_counts['netral']/total*100:.1f}%")
    col3.metric("😊 Positif", f"{sentiment_counts['positif']:.0f}", f"{sentiment_counts['positif']/total*100:.1f}%")

    st.divider()
    st.subheader("Topik Dominan per Sentimen (BERTopic)")
    sentiment_filter = st.selectbox("Pilih sentimen:", ["negatif", "netral", "positif"])
    topic_view = df_topic[df_topic["sentiment"] == sentiment_filter].sort_values(
        "Count", ascending=False
    )[["topic_id", "keywords", "Count"]]
    st.dataframe(topic_view, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Contoh Data Tweet & Prediksi")
    st.dataframe(
        df_pred[["text_raw", "sentiment_label", "sentiment_confidence"]].sample(10, random_state=42),
        use_container_width=True,
        hide_index=True
    )

# ---------- TAB 3: TENTANG METODOLOGI ----------
with tab3:
    st.subheader("📖 Ringkasan Metodologi Penelitian")
    st.markdown("""
    Penelitian ini menganalisis persepsi publik terhadap layanan **TransJakarta**
    melalui data Twitter/X, menggunakan pendekatan **KDD (Knowledge Discovery in Databases)**.

    **Tahapan utama:**
    1. **Pengumpulan data** — 15.738 tweet mentah dikumpulkan, disaring menjadi 14.114 tweet setelah proses cleaning
    2. **Auto-labeling** — menggunakan model Indonesian RoBERTa sebagai pemberi label awal (pseudo-labeling)
    3. **Fine-tuning IndoBERT** — model `indobenchmark/indobert-base-p1` dilatih ulang untuk klasifikasi sentimen 3 kelas
    4. **Pemodelan topik BERTopic** — dijalankan terpisah untuk masing-masing kelas sentimen, menghasilkan 9 topik valid per kelas
    """)

    st.divider()
    st.subheader("📊 Detail Performa Model")
    metrics_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Weighted F1-Score", "Macro F1-Score"],
        "Nilai": ["86.40%", "86.36%", "78.17%"]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("📈 Distribusi Kelas dalam Dataset")
    dist_df = pd.DataFrame({
        "Sentimen": ["Negatif", "Netral", "Positif"],
        "Jumlah": [5674, 7908, 532],
        "Persentase": ["42.56%", "53.98%", "3.46%"]
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
