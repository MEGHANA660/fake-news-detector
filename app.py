import streamlit as st
import pickle
import nltk
import numpy as np
import requests
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

model      = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def get_fake_score(sentence):
    vec   = vectorizer.transform([sentence])
    score = model.decision_function(vec)[0]
    prob  = 1 / (1 + np.exp(-score))
    return round(prob * 100, 2)

def get_fact_check(sentence):
    API_KEY = "AIzaSyDzEzzAZmfQ956MMp4ZD-WCjmg4jH0a2F8"
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={sentence}&key={API_KEY}"
    try:
        res = requests.get(url).json()
        claims = res.get("claims", [])
        if claims:
            claim  = claims[0]
            review = claim.get("claimReview", [{}])[0]
            source = review.get("publisher", {}).get("name", "")
            rating = review.get("textualRating", "")
            link   = review.get("url", "")
            return source, rating, link
    except:
        pass
    return None, None, None

# ---- Page Config ----
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---- Custom CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f0f13; }

.title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.title-block h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff6b6b, #ffa500, #fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.title-block p {
    color: #888;
    font-size: 1rem;
}

.result-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 5px solid;
}
.card-fake    { background: rgba(255,77,77,0.08);  border-color: #ff4d4d; }
.card-susp    { background: rgba(255,165,0,0.08);  border-color: #ffa500; }
.card-real    { background: rgba(40,167,69,0.08);  border-color: #28a745; }

.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.label-fake { color: #ff4d4d; }
.label-susp { color: #ffa500; }
.label-real { color: #28a745; }

.card-score {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}
.card-text {
    color: #ccc;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-top: 6px;
}
.fact-ref {
    background: rgba(74,144,217,0.08);
    border-left: 4px solid #4a90d9;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: -6px;
    margin-bottom: 12px;
    font-size: 0.85rem;
    color: #aaa;
}
.fact-ref a { color: #4a90d9; text-decoration: none; }
.fact-ref a:hover { text-decoration: underline; }

.overall-box {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-top: 24px;
}
.overall-fake { background: rgba(255,77,77,0.1);  border: 1px solid #ff4d4d33; }
.overall-susp { background: rgba(255,165,0,0.1);  border: 1px solid #ffa50033; }
.overall-real { background: rgba(40,167,69,0.1);  border: 1px solid #28a74533; }

.overall-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 2px;
    color: #888;
    margin-bottom: 8px;
}
.overall-score {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
}
.overall-verdict {
    font-size: 1rem;
    margin-top: 6px;
    color: #ccc;
}
.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="title-block">
    <h1>📰 Fake News Detector</h1>
    <p>Paste any news article — each sentence will be analyzed using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---- Input ----
text = st.text_area("", height=200, placeholder="Paste your news article here...")

col1, col2, col3 = st.columns([2,1,2])
with col2:
    analyze = st.button("🔍 Analyze", use_container_width=True)

# ---- Analysis ----
if analyze:
    if text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        sentences = sent_tokenize(text)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📊 Sentence Analysis")

        total_score = 0

        for sentence in sentences:
            score = get_fake_score(sentence)
            total_score += score

            if score >= 70:
                card_class  = "card-fake"
                label_class = "label-fake"
                label_text  = "🔴 LIKELY FAKE"
            elif score >= 45:
                card_class  = "card-susp"
                label_class = "label-susp"
                label_text  = "🟠 SUSPICIOUS"
            else:
                card_class  = "card-real"
                label_class = "label-real"
                label_text  = "🟢 LIKELY REAL"

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="card-label {label_class}">{label_text}</div>
                <div class="card-score {label_class}">{score}% Fake</div>
                <div class="card-text">{sentence}</div>
            </div>
            """, unsafe_allow_html=True)

            # Fact Check
            source, rating, link = get_fact_check(sentence)
            if source and link:
                st.markdown(f"""
                <div class="fact-ref">
                    📰 <b>Fact Check:</b> <b>{source}</b> rates this as 
                    <b style="color:#ffa500">{rating}</b> → 
                    <a href="{link}" target="_blank">Read Full Report ↗</a>
                </div>
                """, unsafe_allow_html=True)

        # ---- Overall Score ----
        avg_score = round(total_score / len(sentences), 2)

        if avg_score >= 70:
            overall_class  = "overall-fake"
            overall_color  = "#ff4d4d"
            overall_emoji  = "⚠️"
            overall_text   = "This article is likely FAKE NEWS"
        elif avg_score >= 45:
            overall_class  = "overall-susp"
            overall_color  = "#ffa500"
            overall_emoji  = "🤔"
            overall_text   = "This article is SUSPICIOUS — verify before sharing"
        else:
            overall_class  = "overall-real"
            overall_color  = "#28a745"
            overall_emoji  = "✅"
            overall_text   = "This article appears to be REAL NEWS"

        st.markdown(f"""
        <div class="overall-box {overall_class}">
            <div class="overall-label">OVERALL FAKE SCORE</div>
            <div class="overall-score" style="color:{overall_color}">{avg_score}%</div>
            <div class="overall-verdict">{overall_emoji} {overall_text}</div>
        </div>
        """, unsafe_allow_html=True)