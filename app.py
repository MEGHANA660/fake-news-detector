import streamlit as st
import pickle
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

model      = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def get_fake_score(sentence):
    vec   = vectorizer.transform([sentence])
    score = model.decision_function(vec)[0]
    prob  = 1 / (1 + np.exp(-score))
    return round(prob * 100, 2)

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste any news article below. Each sentence will be checked for fake content.")

text = st.text_area("Enter News Text Here", height=250, placeholder="Paste your news article here...")

if st.button("🔍 Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        sentences = sent_tokenize(text)
        st.subheader("📊 Results:")
        total_score = 0

        for sentence in sentences:
            score = get_fake_score(sentence)
            total_score += score

            if score >= 70:
                color = "#ff4d4d"
                label = "🔴 LIKELY FAKE"
            elif score >= 45:
                color = "#ffa500"
                label = "🟠 SUSPICIOUS"
            else:
                color = "#28a745"
                label = "🟢 LIKELY REAL"

            st.markdown(
                f"""
                <div style='background-color:{color}22; border-left: 6px solid {color};
                            padding:12px; border-radius:6px; margin-bottom:10px; font-size:15px;'>
                    <b>{label} — {score}% Fake</b><br><br>{sentence}
                </div>
                """,
                unsafe_allow_html=True
            )

        avg_score = round(total_score / len(sentences), 2)
        st.markdown("---")
        st.subheader(f"📌 Overall Article Fake Score: **{avg_score}%**")

        if avg_score >= 70:
            st.error("⚠️ This article is likely FAKE NEWS!")
        elif avg_score >= 45:
            st.warning("🤔 This article is SUSPICIOUS. Verify before sharing.")
        else:
            st.success("✅ This article appears to be REAL NEWS.")
