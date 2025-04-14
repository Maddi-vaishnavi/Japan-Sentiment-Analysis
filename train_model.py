# train_model.py

import re
import requests
import pandas as pd
import nltk
import pickle
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def scrape_wikipedia():
    url = "https://en.wikipedia.org/wiki/Japan"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text(strip=True)

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

def process_data():
    raw_text = scrape_wikipedia()
    cleaned_text = clean_text(raw_text)

    # Sentence-level sentiment
    sentences = sent_tokenize(cleaned_text)
    data = pd.DataFrame({'sentence': sentences})
    data['sentiment'] = data['sentence'].apply(analyze_sentiment)

    # Word-level tokenization for word cloud
    words = word_tokenize(cleaned_text.lower())
    words = [w for w in words if w.isalnum() and w not in stopwords.words('english') and len(w) > 2]

    return data, words

def generate_wordcloud(words):
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def train_and_save_best_model():
    data, words = process_data()
    generate_wordcloud(words)

    # Filter out neutral data
    binary_data = data[data['sentiment'] != 'Neutral'].copy()

    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(binary_data['sentence'])
    y = binary_data['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier()
    }

    best_model = None
    best_f1 = 0.0

    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    # Save model and vectorizer
    with open("best_model_japan.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("tfidf_vectorizer_japan.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    # Optionally save words for debugging or display
    with open("common_words_japan.pkl", "wb") as f:
        pickle.dump(words, f)

    print("Model training complete. Best model, vectorizer, and wordcloud saved.")

if __name__ == "__main__":
    train_and_save_best_model()
