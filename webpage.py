import streamlit as st
import joblib
import numpy as np
import re
import json
import string
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from PIL import Image
from collections import Counter
import pytesseract

# ✅ Set Page Config
st.set_page_config(
    page_title="Cyberbullying & Sentiment Detector",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# Download stopwords if not available
nltk.download("stopwords")

# Configure Tesseract OCR for text extraction from images
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Load models
@st.cache_resource
def load_models():
    try:
        with open("tokenizer.json", "r") as f:
            tokenizer_data = json.load(f)
            
            # Handle different possible tokenizer formats
            if isinstance(tokenizer_data, dict):
                if 'word_index' in tokenizer_data:
                    # Handle Keras tokenizer format
                    tokenizer = tokenizer_data['word_index']
                else:
                    # Assume it's already in the right format
                    tokenizer = tokenizer_data
            else:
                # Fallback: Create an empty dictionary
                tokenizer = {}
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        tokenizer = {}
    
    return {
        # Image-based models
        "rf_model": joblib.load("rf_model.pkl"),
        "knc_model": joblib.load("knc_model.pkl"),
        "etc_model": joblib.load("etc_model.pkl"),
        
        # Text-based models
        "linear_svc_model": joblib.load("svm_model.pkl"),
        "lr_model": joblib.load("lr_model.pkl"),
        "sgd_model": joblib.load("sgd_model.pkl"),
        "mnb_model": joblib.load("mnb_model.pkl"),
        "bilstm_bigru_model": tf.keras.models.load_model("bilstm_bigru_model.h5"),
        
        # Vectorizers and tokenizer
        "vectorizer": joblib.load("vectorizer.pkl"),
        "tfidf_vectorizer": joblib.load("tfidf_vectorizer.pkl"),
        "tokenizer": tokenizer,
        
        # New models
        "tokenizer_2": joblib.load("tokenizer_2.pkl"),
        "roberta_model": joblib.load("roberta_model.pkl")  # Load RoBERTa model using joblib
    }

models = load_models()

# ✅ Sentiment Mapping
sentiment_map = {1: "Negative", 0: "Positive"}

# ✅ Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    
    if len(text.strip()) == 0:
        return None, None, None  # No valid text extracted

    # Safe tokenization for deep learning model
    try:
        tokenizer = models["tokenizer"]
        if isinstance(tokenizer, dict):
            # If it's a dictionary, use get method
            sequence = [tokenizer.get(word, 0) for word in text.split()]
        else:
            # Simple fallback tokenization
            words = list(set(text.split()))
            word_to_index = {word: i+1 for i, word in enumerate(words)}
            sequence = [word_to_index.get(word, 0) for word in text.split()]
        
        # Pad or truncate sequence to fixed length (adjust based on your model's requirements)
        max_len = 100
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence = sequence + [0] * (max_len - len(sequence))
        
        sequence = np.array(sequence).reshape(1, -1)
    except Exception as e:
        st.error(f"Error in tokenization: {e}")
        sequence = np.zeros((1, 100))  # Fallback: empty sequence

    # TF-IDF Vectorization
    try:
        tfidf_vector = models["tfidf_vectorizer"].transform([text]).toarray()
    except Exception as e:
        st.error(f"Error in TF-IDF vectorization: {e}")
        tfidf_vector = np.zeros((1, models["tfidf_vectorizer"].get_feature_names_out().shape[0]))

    # Vectorization for traditional ML models
    try:
        vector = models["vectorizer"].transform([text]).toarray()
    except Exception as e:
        st.error(f"Error in vectorization: {e}")
        vector = np.zeros((1, models["vectorizer"].get_feature_names_out().shape[0]))

    return tfidf_vector, vector, sequence

# ✅ Function to preprocess images for ML models
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((200, 200))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.flatten().reshape(1, -1)  # Flatten to 1D vector
    return image

# ✅ Function to predict sentiment from an image
def predict_sentiment(image):
    extracted_text = pytesseract.image_to_string(image).strip()  # Extract text from image
    processed_image = preprocess_image(image)
    
    # Initialize predictions
    image_predictions = []
    text_predictions = []
    
    # Image-based model predictions
    for model_name in ["rf_model", "knc_model", "etc_model"]:
        try:
            pred = models[model_name].predict(processed_image)[0]
            image_predictions.append(pred)
        except Exception as e:
            st.error(f"Error in {model_name} prediction: {e}")
    
    # Process text if found in image
    if extracted_text:
        try:
            tfidf_vector, vector, sequence = preprocess_text(extracted_text)
            
            if tfidf_vector is not None and vector is not None and sequence is not None:
                # Text-based model predictions with error handling
                try:
                    pred_svc = models["linear_svc_model"].predict(tfidf_vector)[0]
                    text_predictions.append(pred_svc)
                except Exception as e:
                    st.warning(f"SVC model error: {e}")

                try:
                    pred_lr = models["lr_model"].predict(vector)[0]
                    text_predictions.append(pred_lr)
                except Exception as e:
                    st.warning(f"LR model error: {e}")

                try:
                    pred_sgd = models["sgd_model"].predict(vector)[0]
                    text_predictions.append(pred_sgd)
                except Exception as e:
                    st.warning(f"SGD model error: {e}")

                try:
                    pred_mnb = models["mnb_model"].predict(vector)[0]
                    text_predictions.append(pred_mnb)
                except Exception as e:
                    st.warning(f"MNB model error: {e}")

                try:
                    # Ensure the sequence has the right shape for the model
                    pred_bilstm = np.argmax(models["bilstm_bigru_model"].predict(sequence), axis=1)[0]
                    text_predictions.append(pred_bilstm)
                except Exception as e:
                    st.warning(f"BiLSTM model error: {e}")

                    # Predict using RoBERTa model
                    roberta_pred = np.argmax(models["roberta_model"].predict(sequence), axis=1)[0]
                    text_predictions.append(roberta_pred)

        except Exception as e:
            st.error(f"Error processing text from image: {e}")
    
    # Combine predictions
    all_predictions = image_predictions + text_predictions
    if not all_predictions:
        return "Error: No valid predictions!", extracted_text

    # Majority Voting
    final_prediction = Counter(all_predictions).most_common(1)[0][0]
    sentiment = sentiment_map[final_prediction]
    
    return sentiment, extracted_text

# ✅ Function to predict cyberbullying in text
def predict_cyberbullying(text):
    try:
        tfidf_vector, vector, sequence = preprocess_text(text)
        
        if tfidf_vector is None or vector is None or sequence is None:
            return "Invalid Input"
        
        # Predict using text models with error handling
        text_predictions = []
        
        try:
            pred_svc = models["linear_svc_model"].predict(tfidf_vector)[0]
            text_predictions.append(pred_svc)
        except Exception as e:
            st.warning(f"SVC model error: {e}")
            
        try:
            pred_lr = models["lr_model"].predict(vector)[0]
            text_predictions.append(pred_lr)
        except Exception as e:
            st.warning(f"LR model error: {e}")
            
        try:
            pred_sgd = models["sgd_model"].predict(vector)[0]
            text_predictions.append(pred_sgd)
        except Exception as e:
            st.warning(f"SGD model error: {e}")
            
        try:
            pred_mnb = models["mnb_model"].predict(vector)[0]
            text_predictions.append(pred_mnb)
        except Exception as e:
            st.warning(f"MNB model error: {e}")
            
        try:
            pred_bilstm = np.argmax(models["bilstm_bigru_model"].predict(sequence), axis=1)[0]
            text_predictions.append(pred_bilstm)
        except Exception as e:
            st.warning(f"BiLSTM model error: {e}")
        
            # Predict using RoBERTa model
            roberta_pred = np.argmax(models["roberta_model"].predict(sequence), axis=1)[0]
            text_predictions.append(roberta_pred)

        
        if not text_predictions:
            return "Error: All models failed"
        
        # Majority Voting from available predictions
        final_pred = Counter(text_predictions).most_common(1)[0][0]
        
        return "Cyberbullying Detected" if final_pred == 1 else "Non-Bullying Text"
    
    except Exception as e:
        st.error(f"Error in cyberbullying prediction: {e}")
        return "Error in processing"

# ✅ Streamlit UI
st.title("🔍 Cyberbullying & Sentiment Detection App")

# Custom CSS for UI enhancement
st.markdown(
    """
    <style>
    /* Gradient background */
    body {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    /* Custom font for headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Georgia', serif;
        color: #ffffff;
    }
    /* Button styling */
    .stButton button {
        background-color: #ff6f61;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 25px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #ff3b2f;
    }
    /* Text area styling */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #6a11cb;
        padding: 10px;
    }
    /* File uploader styling */
    .stFileUploader label {
        font-size: 16px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar selection
st.sidebar.title("Options")
option = st.sidebar.radio("Choose Input Type", ["Text Data", "Image Data"])

# ✅ Text Data Mode
if option == "Text Data":
    st.header("📜 Cyberbullying Detection")
    user_input = st.text_area("Type your text here:", "", height=200)

    if st.button("Check for Cyberbullying"):
        if user_input.strip():
            with st.spinner("Analyzing text..."):
                result = predict_cyberbullying(user_input)
                if result == "Cyberbullying Detected":
                    st.error(f"🚨 {result}")
                elif result.startswith("Error"):
                    st.warning(f"⚠ {result}")
                else:
                    st.success(f"✅ {result}")
        else:
            st.warning("⚠ Please enter some text.")

# ✅ Image Data Mode
elif option == "Image Data":
    st.header("🖼️ Image Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Sentiment"):
            with st.spinner("Analyzing image..."):
                prediction, extracted_text = predict_sentiment(image)
                
                if prediction.startswith("Error"):
                    st.warning(f"⚠ {prediction}")
                else:
                    if prediction == "Negative":
                        st.error(f"🚨 Predicted Sentiment: **{prediction}**")
                    else:
                        st.success(f"✅ Predicted Sentiment: **{prediction}**")
