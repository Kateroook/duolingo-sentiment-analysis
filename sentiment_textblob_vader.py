import pandas as pd
import nltk
# nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()

# Load preprocessed data
# df = pd.read_csv("tweets_duolingo_preprocessed.csv")
#
# # Fill NaN with empty strings to prevent errors
# df['clean_text_joined'] = df['clean_text_joined'].fillna("")

# # Define sentiment analysis function safely
# def analyze_sentiment_vader(text):
#     if pd.isna(text):
#         return 0, 'neutral'
#     scores = analyzer.polarity_scores(text)
#     compound = scores['compound']
#     if compound >= 0.4:
#         sentiment = 'positive'
#     elif compound <= -0.2:
#         sentiment = 'negative'
#     else:
#         sentiment = 'neutral'
#     return compound, sentiment
#
# # Apply with progress bar
# results = df["clean_text_joined"].apply(analyze_sentiment_vader)
#
# # Split results into two new columns
# df["compound"], df["sentiment"] = zip(*results)
#
# # # Keyword overrides for frustration
# # negative_keywords = ["unusable", "abomination", "bug", "broken", "outage", "hate", "quit", "crash", "energy", "streak lost"]
# # mask = df['clean_text_joined'].str.contains('|'.join(negative_keywords), case=False, na=False)
# # df.loc[mask, 'sentiment'] = 'negative'
#
# # Save results
# df.to_csv("tweets_duolingo_sentiment_vader.csv", index=False)

# print(df[['clean_text_joined', 'compound', 'sentiment']].head())

# Function to analyze sentiment
# def analyze_sentiment_textblob(text):
#     blob = TextBlob(text)
#     polarity = blob.sentiment.polarity  # -1 = negative, 0 = neutral, 1 = positive
#     if polarity > 0.1:
#         sentiment = "positive"
#     elif polarity < -0.1:
#         sentiment = "negative"
#     else:
#         sentiment = "neutral"
#     return polarity, sentiment
# # Apply sentiment analysis
# # df["polarity"], df["sentiment"] = zip(*df["clean_text_joined"].apply(analyze_sentiment_textblob))
# # df.to_csv("tweets_duolingo_sentiment_textblob.csv", index=False)


# Function to classify sentiment
def analyze_sentiment_vader(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']  # overall sentiment score between -1 and 1
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return compound, sentiment

# # Apply sentiment analysis
# df['compound'], df['sentiment'] = zip(*df['clean_text_joined'].apply(analyze_sentiment_vader))
#
#
# # Save results
# df.to_csv("tweets_duolingo_sentiment_vader.csv", index=False)


# print(df[["clean_text_joined", "polarity", "sentiment"]].head())

print("\n🧪 Sample Predictions:")

test_texts = [
    "I love Duolingo! It's so fun and helpful for learning languages.",
    "Lost my streak and the energy system ruins the app.",
    "Just completed my daily lesson on Duolingo.",
    "The new update is annoying and frustrating.",
    "Duolingo helped me learn Spanish quickly. Best app ever!",
    "Come on Duolingo How can this man confidently peruse the glamorous Parisian shopping districts without your immediate help",
    "Is that why my Duolingo stopped working",
    "Duolingo episode would have made me want to kill myself if this was a finale to a show I was genuinely invested in and was not watching for the LOLs That is all I will say about it",
    "Im glad i know what this means my Duolingo lessons paying off",
    "Amazon is cloud services unit Amazon Web Services AWS has been hit by an outage causing connectivity issues for many companies around the world The services disrupted include several popular websites and apps including Fortnite Snapchat and Duolingo",
    "My Duolingo streak better not die I m scared of that bird",
    "Duolingo actually helped me speak and read espa ol a lot better",
    "I study italian on duolingo and it full of and diversity stuff Just unbelievable",
    "for your information Snapchat Fortnite Clash Royale Roblox Alexa Amazon Amazon Prime Video all Amazon devices Duolingo Canva The NY Times Apple TV PUBG AND me are all down rn lol",
    "Duolingo is gracious and calling it a maintenance break"

]

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Preprocess new text for prediction"""
    # Clean
    text = text.lower()
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens


for text in test_texts:
    tokens = preprocess_text(text)
    prediction = analyze_sentiment_vader(text)
    print(f"\nText: {text}")
    print(f"Prediction: {prediction}")
