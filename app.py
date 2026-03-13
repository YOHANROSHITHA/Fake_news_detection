import base64
from pathlib import Path
import pickle
import re
import string

import streamlit as st
from nltk.corpus import stopwords

# Load Model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Persisted threshold tuned on validation; fall back to 0.5 if absent
DEFAULT_THRESHOLD = float(getattr(model, "best_threshold_", 0.5))
DEFAULT_THRESHOLD = min(max(DEFAULT_THRESHOLD, 0.1), 0.9)

CLASS_INDEX = {cls: idx for idx, cls in enumerate(getattr(model, "classes_", [0, 1]))}
REAL_IDX = CLASS_INDEX.get(1, 1 if len(CLASS_INDEX) > 1 else 0)

stop_words = set(stopwords.words("english"))


def set_background(image_path: str = "image1.png") -> None:
    """Set a blurred background image using base64 CSS."""
    try:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    except FileNotFoundError:
        return

    css = f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: url("data:image/png;base64,{b64}") center/cover no-repeat;
        z-index: -2;
    }}
    .stApp {{
        background: rgba(255, 255, 255, 0.80);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def clean_text(text):
    """Mirror training-time cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# UI
st.title("Fake News Detection System")
st.write("Enter news article text below:")

set_background()

news = st.text_area("News Text")
threshold = st.slider(
    "Decision threshold (real if probability >= threshold)",
    0.1,
    0.9,
    DEFAULT_THRESHOLD,
    0.05,
)

if st.button("Check News"):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])

    probability = model.predict_proba(vector)[0]
    prob_real = probability[REAL_IDX] * 100
    prob_fake = probability[1 - REAL_IDX] * 100

    label_real = prob_real / 100 >= threshold

    if label_real:
        st.success(f"Real News (P(real): {prob_real:.2f}%, P(fake): {prob_fake:.2f}%)")
    else:
        st.error(f"Fake News (P(fake): {prob_fake:.2f}%, P(real): {prob_real:.2f}%)")