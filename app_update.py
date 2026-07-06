import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.preprocessing import preprocess_for_sentiment

# ============ KONFIGURASI ============
MODEL_DIR = "zacheldy/transjakarta-indobert-sentiment"  # Hugging Face Hub
MAX_LENGTH = 128
ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}
DATASET_PATH = "data/dataset_05_indobert_predicted_from_roberta_label.csv"
TOPIC_PATH = "data/bertopic_topic_summary_all_sentiments_roberta_max10.csv"

st.set_page_config(page_title="Analisis Sentimen TransJakarta", layout="wide")


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


def predict_sentiment(text_raw: str):
    text_clean = preprocess_for_sentiment(text_raw)
    inputs = tokenizer(
        text_clean,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], float(probs[pred_id]), text_clean, probs


# ============ UI ============
st.title("🚌 Analisis Sentimen & Topik Layanan TransJakarta")
st.caption("Skripsi: Analisis Sentimen dan Pemodelan Topik terhadap Layanan TransJakarta di Twitter/X Menggunakan IndoBERT dan BERTopic")

tab1, tab2 = st.tabs(["🔍 Coba Sendiri", "📊 Dashboard Hasil Riset"])

with tab1:
    st.subheader("Coba Prediksi Sentimen")
    user_input = st.text_area(
        "Masukkan teks/tweet tentang TransJakarta:",
        placeholder="Contoh: busnya lama banget, udah nunggu 30 menit di halte"
    )

    if st.button("Prediksi Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            label, confidence, text_clean, probs = predict_sentiment(user_input)
            emoji_map = {"negatif": "😠", "netral": "😐", "positif": "😊"}
            st.markdown(f"### Hasil: {emoji_map[label]} **{label.upper()}**")
            st.metric("Confidence Score", f"{confidence*100:.2f}%")

            with st.expander("Detail teknis (teks setelah preprocessing)"):
                st.code(text_clean)
                st.write({
                    "negatif": f"{probs[0]*100:.2f}%",
                    "netral": f"{probs[1]*100:.2f}%",
                    "positif": f"{probs[2]*100:.2f}%",
                })

with tab2:
    st.subheader("Distribusi Sentimen (14.114 tweet)")
    sentiment_counts = df_pred["sentiment_label"].value_counts().reindex(
        ["negatif", "netral", "positif"]
    ).fillna(0)
    st.bar_chart(sentiment_counts)

    col1, col2, col3 = st.columns(3)
    total = len(df_pred)
    col1.metric("Negatif", f"{sentiment_counts['negatif']:.0f}", f"{sentiment_counts['negatif']/total*100:.1f}%")
    col2.metric("Netral", f"{sentiment_counts['netral']:.0f}", f"{sentiment_counts['netral']/total*100:.1f}%")
    col3.metric("Positif", f"{sentiment_counts['positif']:.0f}", f"{sentiment_counts['positif']/total*100:.1f}%")

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
