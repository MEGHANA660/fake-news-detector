import streamlit as st
import pickle
import nltk
import numpy as np
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

nltk.download('punkt')
nltk.download('punkt_tab')

# ---- Train and save model if not already saved ----
@st.cache_resource
def load_model():
    if not os.path.exists("model/model.pkl"):
        st.info("Training model for first time... please wait!")
        real = pd.read_csv("data/archive/True.csv")
        fake = pd.read_csv("data/archive/Fake.csv")
        real["label"] = 0
        fake["label"] = 1
        df = pd.concat([real, fake]).sample(frac=1).reset_index(drop=True)
        df["text"] = df["title"] + " " + df["text"]

        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )
        vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        model = PassiveAggressiveClassifier(max_iter=50)
        model.fit(X_train_vec, y_train)
        os.makedirs("model", exist_ok=True)
        pickle.dump(model, open("model/model.pkl", "wb"))
        pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

    model = pickle.load(open("model/model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---- Function to get fake % ----
def get_fake_score(sentence):
    vec   = vectorizer.transform([sentence])
    score = model.decision_function(vec)[0]
    prob  = 1 / (1 + np.exp(-score))
    return round(prob * 100, 2)

# ---- UI ----
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