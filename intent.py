from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

model = joblib.load("random_forest_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

app = FastAPI(title="Terms & Conditions Risk API")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the Terms & Conditions Risk API!"}

@app.post("/predict/")
def predict_risk(input_text: TextInput):
    cleaned_text = preprocess_text(input_text.text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    
    result = "Risky" if prediction[0] == 1 else "Safe"
    return {"text": input_text.text, "prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)