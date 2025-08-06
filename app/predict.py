
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import string
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy
from transformers import DistilBertTokenizerFast, TFDistilBertModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

MAX_LEN = 128


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm.keras")
BERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "bert.keras")
LSTM_TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "token", "lstm_tokenizer.pkl")
BERT_TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "token", "bert_tokenizer")
BERT_THRESHOLDS_PATH = os.path.join(BASE_DIR, "models", "thresholds", "thresholds_bert.npy")
LSTM_THRESHOLDS_PATH = os.path.join(BASE_DIR, "models", "thresholds", "thresholds_lstm.npy")


emo_list = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
negations = {"not", "no", "never", "n't"}
stop_words = stop_words.difference(negations)

def clean(text):
    text = text.lower()
    text = re.sub(r"["                 
                  "\U0001F600-\U0001F64F"
                  "\U0001F300-\U0001F5FF"
                  "\U0001F680-\U0001F6FF"
                  "\U0001F1E0-\U0001F1FF"
                  "\U00002700-\U000027BF"
                  "\U000024C2-\U0001F251"
                  "]+", '', text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

bert_model = tf.keras.models.load_model(
    BERT_MODEL_PATH,
    custom_objects={"TFDistilBertModel": TFDistilBertModel}
)
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_TOKENIZER_PATH)

def loss_fn(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

lstm_model = load_model(LSTM_MODEL_PATH, custom_objects={'loss_fn': loss_fn})
with open(LSTM_TOKENIZER_PATH, 'rb') as f:
    lstm_tokenizer = pickle.load(f)

thresholds_bert = np.load(BERT_THRESHOLDS_PATH)
thresholds_lstm = np.load(LSTM_THRESHOLDS_PATH)

def predict_bert(text):
    inputs = bert_tokenizer([text], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="tf")
    probs = bert_model.predict({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, verbose=0)[0]
    pred = (probs > thresholds_bert).astype(int)
    results = [(emo, round(float(prob), 3)) for emo, prob, p in zip(emo_list, probs, pred) if p == 1]
    return results if results else [("other", round(float(max(probs)), 3))]

def predict_lstm(text):
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = lstm_model.predict(padded, verbose=0)[0]
    pred = (probs > thresholds_lstm).astype(int)
    results = [(emo, round(float(prob), 3)) for emo, prob, p in zip(emo_list, probs, pred) if p == 1]
    return results if results else [("other", round(float(max(probs)), 3))]
