import pandas as pd
import re
import spacy
import html
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ----------------------------
# Load the dataset
# ----------------------------
df = pd.read_csv("tweets_duolingo.csv")

# ----------------------------
# Define text cleaning function
# ----------------------------
def remove_noise(text):
    text = str(text)

    # Decode HTML entities like &amp;, &lt;, &gt;, etc.
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove punctuation, numbers, special characters
    text = re.sub(r"[^A-Za-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
df['text_cleaned'] = df['text'].apply(remove_noise)

# ----------------------------
# Normalization (Lower Case + Stopword Removal + Lemmatization)
# ----------------------------

def normalize_spacy(text):
    # Convert to lowercase
    text = text.lower()
    # Process text through spaCy pipeline
    doc = nlp(text)

    # Lemmatize, remove stopwords and short tokens
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and len(token.text) > 2 and token.is_alpha
    ]

    return tokens

df['tokens'] = df['text_cleaned'].apply(normalize_spacy)
df['clean_text_joined'] = df['tokens'].apply(lambda x: ' '.join(x))

# ----------------------------
# Save preprocessed data
# ----------------------------
df.to_csv("tweets_duolingo_preprocessed_spacy.csv", index=False, encoding='utf-8-sig')

print("✅ Data preprocessing complete.")
print(df[['text', 'text_cleaned', 'tokens']].head(5))