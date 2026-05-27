import streamlit as st
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================
# DOWNLOAD NLTK DATA
# ============================================

nltk.download('stopwords')

# ============================================
# LOAD MODEL AND FILES
# ============================================

model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ============================================
# SETTINGS
# ============================================

max_length = 100

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    layout="wide"
)

# ============================================
# HEADER SECTION
# ============================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Bidirectional LSTM"
)

# ============================================
# ABOUT PROJECT SECTION
# ============================================

st.markdown("## 📘 About the Project")

st.write("""
This AI-powered application analyzes emotional sentiment from user text using Natural Language Processing (NLP) and Deep Learning.

### Importance of Emotional AI
- Detects emotional patterns
- Helps identify mental stress
- Supports emotional wellness monitoring
- Assists counselors with early intervention

### NLP Applications
- Sentiment analysis
- Chatbots
- Emotion recognition
- Mental wellness systems

### Role of RNN/LSTM
Bidirectional LSTM processes text sequentially while remembering previous context from both directions, improving emotional understanding.
""")

# ============================================
# USER INPUT SECTION
# ============================================

st.markdown("## ✍️ Enter Your Thoughts")

st.write("### Example Sentences")
st.write("- I feel emotionally exhausted and lonely")
st.write("- I am very happy today")
st.write("- I feel anxious about my future")
st.write("- Nobody understands what I am going through")

user_input = st.text_area(
    "User Text",
    placeholder="Enter your thoughts or feelings here...",
    height=180
)

# ============================================
# TEXT PREPROCESSING
# ============================================

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation only
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

# ============================================
# EMOTIONAL GUIDANCE FUNCTION
# ============================================

def emotional_guidance(emotion):

    emotion = emotion.lower()

    if emotion in ['depression', 'sadness']:

        return """
        💙 Try talking with someone you trust.
        
        🌿 Take short breaks and practice self-care.
        
        ☀️ Small positive activities can improve mood gradually.
        """

    elif emotion in ['anxiety', 'stress']:

        return """
        🌸 Practice deep breathing exercises.
        
        🚶 Take a short walk and relax your mind.
        
        🧘 Focus on one task at a time.
        """

    elif emotion in ['happy', 'joy']:

        return """
        🎉 Keep spreading positivity.
        
        😊 Maintain healthy routines and hobbies.
        
        💪 Continue activities that make you feel motivated.
        """

    else:

        return """
        🌼 Maintain emotional balance with healthy habits.
        
        📚 Spend time on relaxing activities.
        
        💧 Stay hydrated and take proper rest.
        """

# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        # ============================================
        # PREPROCESS TEXT
        # ============================================

        cleaned_text = preprocess_text(user_input)

        # ============================================
        # TOKENIZATION
        # ============================================

        sequence = tokenizer.texts_to_sequences([cleaned_text])

        # ============================================
        # PADDING
        # ============================================

        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_length,
            padding='post'
        )

        # ============================================
        # MODEL PREDICTION
        # ============================================

        prediction = model.predict(padded_sequence)

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction)

        predicted_emotion = encoder.inverse_transform(
            [predicted_index]
        )[0]

        # ============================================
        # PREDICTION OUTPUT
        # ============================================

        st.markdown("## 📊 Prediction Result")

        st.success(
            f"Emotion Detected: {predicted_emotion}"
        )

        st.info(
            f"Confidence Score: {round(confidence * 100, 2)}%"
        )

        # Emotional Status

        if confidence >= 0.85:

            st.write(
                "Emotional Status: Strong emotional signal detected"
            )

        elif confidence >= 0.60:

            st.write(
                "Emotional Status: Moderate emotional signal detected"
            )

        else:

            st.write(
                "Emotional Status: Mixed emotional pattern detected"
            )

        # ============================================
        # VISUALIZATION SECTION
        # ============================================

        st.markdown("## 📈 Emotion Probability Chart")

        emotions = encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(emotions, probabilities)

        ax.set_xlabel("Emotion")

        ax.set_ylabel("Confidence")

        ax.set_title("Emotion Prediction Confidence")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        # ============================================
        # WELLNESS GUIDANCE SECTION
        # ============================================

        st.markdown("## 💡 Emotional Wellness Guidance")

        guidance = emotional_guidance(predicted_emotion)

        st.write(guidance)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.caption(
    "AI-Based Mental Health Sentiment Monitoring System using Deep Learning"
)