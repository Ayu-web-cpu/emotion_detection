# predict_emotion.py

import pickle
import re
import nltk

# Download stopwords if needed
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load model, vectorizer, and encoder
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Predict emotion from user input
def predict_emotion(text):
    text = preprocess(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

# Input loop
while True:
    user_input = input("\nEnter your message (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    emotion = predict_emotion(user_input)
    print(f"Predicted Emotion: {emotion}")
