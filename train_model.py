# train_model.py

import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load dataset (from same directory)
df = pd.read_csv("train.txt", sep=';', header=None, names=["text", "emotion"])

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['text'] = df['text'].apply(preprocess)

# Print class distribution
print("\n📊 Emotion Class Distribution:\n")
print(df['emotion'].value_counts())

# Features and labels
X = df['text']
y = df['emotion']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Vectorize with TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

# Train Logistic Regression with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_vec, y_encoded)

# Save model and components
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅ Model, vectorizer, and encoder saved successfully.")




