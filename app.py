import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER safely (works on Streamlit Cloud)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

st.set_page_config(page_title="Product Feedback Dashboard", layout="wide")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_sentiment_scores(texts):
    sia = SentimentIntensityAnalyzer()
    scores = []
    labels = []
    for t in texts:
        s = sia.polarity_scores(t)["compound"]
        scores.append(s)
        if s >= 0.05:
            labels.append("Positive")
        elif s <= -0.05:
            labels.append("Negative")
        else:
            labels.append("Neutral")
    return scores, labels

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("feedback.csv")

st.subheader("📌 Raw Feedback Data")
st.dataframe(df, use_container_width=True)

# Find feedback column
feedback_col = None
for col in df.columns:
    if "feedback" in col.lower() or "review" in col.lower() or "comment" in col.lower():
        feedback_col = col
        break

if feedback_col is None:
    st.error("❌ No feedback column found. Please name your column as 'feedback'.")
    st.stop()

df["clean_feedback"] = df[feedback_col].apply(clean_text)

# Sentiment
scores, labels = get_sentiment_scores(df["clean_feedback"])
df["sentiment_score"] = scores
df["sentiment_label"] = labels

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("😊 Sentiment Distribution")
    sentiment_counts = df["sentiment_label"].value_counts()
    st.bar_chart(sentiment_counts)

with col2:
    st.subheader("📈 Sentiment Score Summary")
    st.write(df["sentiment_score"].describe())

# Keyword extraction (simple)
stopwords = set([
    "the","is","and","a","to","of","in","it","for","on","this","that","very","too",
    "i","my","we","you","app","please","add","need","want","with","when","try"
])

all_words = []
for text in df["clean_feedback"]:
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    all_words.extend(words)

word_counts = Counter(all_words)
top_keywords = word_counts.most_common(15)

st.subheader("🔥 Top Keywords (Pain Points / Requests)")
keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Count"])
st.dataframe(keywords_df, use_container_width=True)

# Basic product insights rules
def classify_feedback(text):
    if any(k in text for k in ["crash", "bug", "error", "issue"]):
        return "Bug/Crash"
    if any(k in text for k in ["slow", "load", "lag"]):
        return "Performance"
    if any(k in text for k in ["dark mode", "export", "feature", "reminder", "notifications"]):
        return "Feature Request"
    if any(k in text for k in ["support", "helpful"]):
        return "Customer Support"
    if any(k in text for k in ["ads", "annoying"]):
        return "User Experience"
    return "General"

df["category"] = df["clean_feedback"].apply(classify_feedback)

st.subheader("🧠 Feedback Category Breakdown")
category_counts = df["category"].value_counts()
st.bar_chart(category_counts)

# Roadmap suggestions (simple)
st.subheader("🚀 Roadmap Recommendations (Auto)")

recommendations = []
for cat, count in category_counts.items():
    if cat == "Bug/Crash":
        recommendations.append(("Fix crashes & login bugs", count))
    elif cat == "Performance":
        recommendations.append(("Improve app speed & loading time", count))
    elif cat == "Feature Request":
        recommendations.append(("Build requested features (dark mode, export, reminders)", count))
    elif cat == "User Experience":
        recommendations.append(("Reduce ads & improve UX", count))
    elif cat == "Customer Support":
        recommendations.append(("Maintain/improve support quality", count))

rec_df = pd.DataFrame(recommendations, columns=["Recommendation", "Mentions"])
rec_df = rec_df.sort_values(by="Mentions", ascending=False)

st.dataframe(rec_df, use_container_width=True)

# Download insights
st.subheader("⬇️ Download Analyzed Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV Report", data=csv, file_name="feedback_analysis_report.csv", mime="text/csv")
