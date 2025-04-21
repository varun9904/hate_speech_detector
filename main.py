from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https\S+|www\S+http\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_text(request: TextRequest):
    raw_text = request.text
    cleaned_text = clean_text(raw_text)
    
    input_vector = vectorizer.transform([cleaned_text])
    
    predicted_label = model.predict(input_vector)[0]
    predicted_probabilities = model.predict_proba(input_vector)[0]
    
    prediction_text = "Hate Speech" if predicted_label == 1 else "Safe Speech"
    
    return {
        "prediction": prediction_text,
        "probabilities": {
            "safe": float(predicted_probabilities[0]),
            "hate": float(predicted_probabilities[1])
        }
    }
