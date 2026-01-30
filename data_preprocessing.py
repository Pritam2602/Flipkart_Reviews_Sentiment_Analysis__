import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

output_dir = r"reviews_data\processed"
os.makedirs(output_dir, exist_ok=True)

COLUMN_MAPPING = {
    # Ratings
    "ratings": "rating",
    "reviewer_rating": "rating",

    # Review title
    "review_title": "review_title",
    "review title": "review_title",

    # Review text
    "review_text": "review_text",
    "review text": "review_text"
}

def normalize_and_map_columns(df):
    # normalize
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # map to standard schema
    df = df.rename(columns=COLUMN_MAPPING)

    return df


def load_data():
    badminton = pd.read_csv(r"D:\Flipkart_Reviews_Sentiment_Analysis__\reviews_data\reviews_badminton\data.csv")
    tawa = pd.read_csv(r"D:\Flipkart_Reviews_Sentiment_Analysis__\reviews_data\reviews_tawa\data.csv")
    tea = pd.read_csv(r"D:\Flipkart_Reviews_Sentiment_Analysis__\reviews_data\reviews_tea\data.csv")

    badminton = normalize_and_map_columns(badminton)
    tawa = normalize_and_map_columns(tawa)
    tea = normalize_and_map_columns(tea)

    badminton["product"] = "badminton"
    tawa["product"] = "tawa"
    tea["product"] = "tea"

    return badminton, tawa, tea

def combine_text(df):
    df["text"] = (
        df["review_title"].fillna("") + " " +
        df["review_text"].fillna("")
    )
    return df

def create_sentiment(df):
    df = df.copy()
    def label(r):
        if r >= 4:
            return 1
        elif r <= 2:
            return 0
        else:
            return None

    df["sentiment"] = df["rating"].apply(label)
    df = df.dropna(subset=["sentiment"]).copy()

    return df

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

def preprocess_all():
    badminton, tawa, tea = load_data()

    dfs = []
    for df in [badminton, tawa, tea]:
        df = combine_text(df)
        df = create_sentiment(df)
        df = df.copy()
        df["clean_text"] = df["text"].apply(clean_text)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    return final_df[["clean_text", "sentiment", "product"]]


if __name__ == "__main__":
    df = preprocess_all()
    print(df.head())
    print(df["sentiment"].value_counts())
    print(df["product"].value_counts())
    df.to_csv(
    os.path.join(output_dir, "cleaned_data.csv"),
    index=False)

