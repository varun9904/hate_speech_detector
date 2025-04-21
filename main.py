import os
import nltk

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

for pkg in ["stopwords", "punkt", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)
        
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import time
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import speech_recognition as sr

app = FastAPI()

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
    processed_text = clean_text(raw_text)
    
    input_vector = vectorizer.transform([processed_text])
    predicted_label = model.predict(input_vector)[0]
    predicted_probabilities = model.predict_proba(input_vector)[0]
    
    prediction_text = "Hate Speech" if predicted_label == 1 else "Safe Speech"
    
    return {
        "cleaned_text": processed_text,
        "prediction": prediction_text,
        "probabilities": {
            "safe": float(predicted_probabilities[0]),
            "hate": float(predicted_probabilities[1])
        }
    }

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        extension = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            temp_file_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(temp_file_path)
            wav_path = temp_file_path + ".wav"
            audio.export(wav_path, format="wav")
            os.remove(temp_file_path)
        except Exception as e:
            os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Could not process audio file: {str(e)}")

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                transcribed_text = recognizer.recognize_google(audio_data)
        except Exception as e:
            os.remove(wav_path)
            raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        processed_transcribed_text = clean_text(transcribed_text)
        input_vector = vectorizer.transform([processed_transcribed_text])
        predicted_label = model.predict(input_vector)[0]
        predicted_probabilities = model.predict_proba(input_vector)[0]
        prediction_text = "Hate Speech" if predicted_label == 1 else "Safe Speech"
    
        return {
            "original_transcription": transcribed_text,
            "cleaned_text": processed_transcribed_text,
            "prediction": prediction_text,
            "probabilities": {
                "safe": float(predicted_probabilities[0]),
                "hate": float(predicted_probabilities[1])
            }
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            time.sleep(1)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except PermissionError as e:
            return {"error": f"Permission error while removing the file: {str(e)}"}
