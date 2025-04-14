import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# Page config
st.set_page_config(page_title="Japan Wiki Sentiment Analyzer", layout="wide", page_icon="🇯🇵")

# Load resources
with open("best_model_japan.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer_japan.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App Header
st.title("🇯🇵 Japan Wikipedia Sentiment Analyzer")
st.markdown(
    """
    Analyze the sentiment of text related to Japan's Wikipedia page.  
    Uses NLP + ML pipeline with **TF-IDF**, **SMOTE**, and **sklearn models**.
    """
)

st.sidebar.header("🔍 Predict Sentiment")
user_input = st.sidebar.text_area("Enter your sentence below:", height=80)

if st.sidebar.button("🎯 Predict Sentiment"):
    if user_input.strip() == "":
        st.sidebar.warning("Please enter a sentence to analyze.")
    else:
        # Sentiment Analysis
        def get_sentiment(text):
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                return "Positive"
            elif polarity == 0:
                return "Neutral"
            else:
                return "Negative"

        def predict_sentiment(text):
            sentiment = get_sentiment(text)
            vectorized = vectorizer.transform([text])
            if sentiment == "Neutral":
                return sentiment, {"Neutral": 1.0, "Positive": 0.0, "Negative": 0.0}
            prediction = model.predict_proba(vectorized)[0]
            return sentiment, {"Neutral": 0.0, "Positive": prediction[1], "Negative": prediction[0]}

        predicted_sentiment, result = predict_sentiment(user_input)

        # Output: Sentiment Label
        st.sidebar.success(f"Predicted Sentiment: **{predicted_sentiment}**")

        # Prediction Details Table
        st.subheader("📋 Prediction Details")
        result_df = pd.DataFrame(result.items(), columns=["Sentiment", "Probability"])
        st.dataframe(result_df.style.format({"Probability": "{:.2%}"}))

        # Bar Chart
        st.subheader("📊 Prediction Probability")
        fig, ax = plt.subplots()
        sns.barplot(x="Sentiment", y="Probability", data=result_df, palette="Set2", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Sentiment Probability")
        st.pyplot(fig)

# Word Cloud
st.subheader("☁️ Word Cloud from Wikipedia Page on Japan")
st.image("japan_wordcloud.png", use_column_width=True)

st.markdown("---")
st.caption("Developed by a Data Science enthusiast 🚀")
