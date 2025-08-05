from fastapi import FastAPI
from pydantic import BaseModel
from predict import clean, predict_bert, predict_lstm

app = FastAPI(title="Emotion Classifier API")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def get_prediction(data: TextInput):
    raw = data.text
    cleaned = clean(raw)

    bert_output = predict_bert(cleaned)
    lstm_output = predict_lstm(cleaned)

    return {
        "input": raw,
        "cleaned": cleaned,
        "bert_predictions": bert_output,
        "lstm_predictions": lstm_output
    }
