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

model_filename = 'best_model.pkl'
vectorizer_filename = 'vectorizer.pkl'

model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

print("Model and vectorizer loaded successfully with joblib!\n")

while True:
    user_input = input("Enter a comment to classify (or type 'q' to quit): ")
    if user_input.lower() == 'q':
        print("Exiting program.")
        break

    clean_input = clean_text(user_input)
    
    input_vector = vectorizer.transform([clean_input])
    
    predicted_label = model.predict(input_vector)[0]
    predicted_probabilities = model.predict_proba(input_vector)[0]
    
    if predicted_label == 1:
        prediction_text = "Hate Speech"
    else:
        prediction_text = "Safe"
    
    print("\nPrediction:", prediction_text)
    print("Predicted Probabilities (Safe, Hate): ({:.2f}, {:.2f})\n".format(
        predicted_probabilities[0], predicted_probabilities[1]
    ))
