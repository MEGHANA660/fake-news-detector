import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---- Load Data ----
print("Loading data...")
real = pd.read_csv("data/archive/True.csv")
fake = pd.read_csv("data/archive/Fake.csv")

# ---- Label the data ----
# 0 = Real News, 1 = Fake News
real["label"] = 0
fake["label"] = 1

# ---- Combine both into one big dataset ----
df = pd.concat([real, fake])
df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

# ---- Use title + text as input ----
df["text"] = df["title"] + " " + df["text"]

# ---- Split into Training and Testing data ----
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

print("Training model... please wait...")

# ---- Convert text to numbers (TF-IDF) ----
vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ---- Train the ML Model ----
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# ---- Check Accuracy ----
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# ---- Save the trained model ----
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully in /model folder!")