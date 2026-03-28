# 📰 Fake News Detector

A Machine Learning web application that detects fake news **sentence by sentence** and shows the fake percentage for each line along with real news references.

---

## 🚀 Demo

Paste any news article → Each sentence gets analyzed → See which lines are fake with % score + news source references!

---

## ✨ Features

- 🔍 **Sentence-by-sentence analysis** — detects which exact line is fake
- 📊 **Fake percentage score** for each sentence
- 🔴🟠🟢 **Color coded results** — Red (Fake), Orange (Suspicious), Green (Real)
- 📰 **Real-time news references** — shows related sources from BBC, CNN, Reuters etc.
- 📌 **Overall article fake score** with final verdict
- ⚡ **99.61% model accuracy**

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | ML model training |
| TF-IDF Vectorizer | Text to numbers conversion |
| Passive Aggressive Classifier | Fake news classification |
| NLTK | Sentence tokenization |
| Streamlit | Web application UI |
| NewsAPI | Real-time news references |
| Pickle | Model saving and loading |

---

## 📁 Project Structure

```
fake_news_detector/
│
├── data/
│   └── archive/
│       ├── True.csv        ← Real news dataset
│       └── Fake.csv        ← Fake news dataset
│
├── model/
│   ├── model.pkl           ← Trained ML model
│   └── vectorizer.pkl      ← TF-IDF vectorizer
│
├── app.py                  ← Main Streamlit web app
├── train_model.py          ← Model training script
├── tunnel.py               ← ngrok tunnel for sharing
├── requirements.txt        ← Required libraries
└── README.md
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/MEGHANA660/fake-news-detector.git
cd fake-news-detector
```

### Step 2 — Install dependencies
```bash
pip install pandas scikit-learn nltk streamlit numpy requests
```

### Step 3 — Download Dataset
Download the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place `True.csv` and `Fake.csv` inside `data/archive/` folder.

### Step 4 — Train the model
```bash
python train_model.py
```
This will generate `model.pkl` and `vectorizer.pkl` inside the `model/` folder.

### Step 5 — Run the app
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 🧠 How It Works

1. News article is split into individual sentences using NLTK
2. Each sentence is converted to numbers using **TF-IDF Vectorizer**
3. The **Passive Aggressive Classifier** predicts if each sentence is fake or real
4. A sigmoid function converts the raw score into a **0-100% fake probability**
5. **NewsAPI** searches for related real news sources for reference
6. Overall article score is calculated as average of all sentence scores

---

## 📊 Model Details

| Detail | Value |
|---|---|
| Dataset | ISOT Fake News Dataset |
| Total Articles | 44,898 |
| Real News | 21,417 (Reuters) |
| Fake News | 23,481 |
| Algorithm | Passive Aggressive Classifier |
| Feature Extraction | TF-IDF (max_df=0.7) |
| Train/Test Split | 80% / 20% |
| Model Accuracy | **99.61%** |

---

## ⚠️ Limitations

- Model is trained on **American English news** — may show higher fake scores for Indian regional news due to dataset bias
- **Future Enhancement**: Train on multilingual datasets or use **XLM-RoBERTa** for 100+ language support

---

## 👩‍💻 Author

**Meghana K**  
Mini Project — Fake News Detection using Machine Learning

