import pandas as pd
import random
import nltk
import pickle
import html
import re
from nltk.corpus import stopwords, opinion_lexicon, wordnet
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, extract_bigram_feats, mark_negation
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.classify.util import accuracy

# ----------------------------
# Setup (uncomment on first run)
# ----------------------------
# nltk.download('stopwords')
# nltk.download('opinion_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

df = pd.read_csv("tweets_to_train.csv")

# ----------------------------
# 1️⃣ Cleaning
# ----------------------------
def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# df['text_cleaned'] = df['text_cleaned'].apply(clean_text)

# ----------------------------
# 2️⃣ Lemmatization + POS tagging + stopwords
# ----------------------------
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nor','down'}
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    tag = tag[0].upper()
    return {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
        if word.isalpha() and word not in stop_words
    ]
    return lemmatized

df['tokens'] = df['text_cleaned'].apply(preprocess_text)
# ----------------------------
# 3️⃣ Prepare dataset safely
# ----------------------------
if 'tokens' not in df.columns or df['tokens'].isnull().any():
    print("⚠️ Regenerating tokens from text column...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    df['tokens'] = df['text_cleaned'].apply(preprocess_text)

# Drop rows with invalid tokens or labels
df = df[df['tokens'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
df = df.dropna(subset=['tokens', 'label'])

dataset = [(tokens, label) for tokens, label in zip(df['tokens'], df['label'])]

random.seed(42)
random.shuffle(dataset)

split = int(len(dataset) * 0.8)
train_data, test_data = dataset[:split], dataset[split:]

print(f"📊 Total samples: {len(dataset)}")
print(f"Training: {len(train_data)}, Testing: {len(test_data)}")

# ----------------------------
# 4️⃣ SentimentAnalyzer setup
# ----------------------------
sentim_analyzer = SentimentAnalyzer()

# Apply negation marking (so “not good” is kept as one concept)
train_data = [(mark_negation(doc), label) for doc, label in train_data]
test_data = [(mark_negation(doc), label) for doc, label in test_data]

# Collect all words
all_words = sentim_analyzer.all_words([doc for doc, _ in train_data])

# Select most frequent unigrams (to reduce noise)
freq_dist = nltk.FreqDist(all_words)
most_common = [w for w, f in freq_dist.items() if f >= 2]  # only words appearing >=2 times

print(f"✅ Using {len(most_common)} unigram features")

# Build bigrams
bigram_finder = BigramCollocationFinder.from_words(all_words)
bigram_feats = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 200)

# Add feature extractors
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=most_common)
sentim_analyzer.add_feat_extractor(extract_bigram_feats, bigrams=bigram_feats)

# Add lexicon-based features
pos_words = set(opinion_lexicon.positive())
neg_words = set(opinion_lexicon.negative())

def extract_sentiment_lexicon_feats(tokens):
    feats = {
        'has_positive': any(t in pos_words for t in tokens),
        'has_negative': any(t in neg_words for t in tokens),
        'pos_count': sum(t in pos_words for t in tokens),
        'neg_count': sum(t in neg_words for t in tokens),
    }
    return feats

sentim_analyzer.add_feat_extractor(extract_sentiment_lexicon_feats)

# ----------------------------
# 5️⃣ Train classifier
# ----------------------------
train_features = sentim_analyzer.apply_features(train_data)
test_features = sentim_analyzer.apply_features(test_data)

print("\n🤖 Training Naive Bayes Classifier...")
trainer = nltk.classify.NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, train_features)

# ----------------------------
# 6️⃣ Evaluate
# ----------------------------
print("\n📈 Model Evaluation:")
train_acc = accuracy(classifier, train_features)
test_acc = accuracy(classifier, test_features)
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

classifier.show_most_informative_features(15)

# ----------------------------
# 7️⃣ Save model
# ----------------------------
with open("duolingo_sentiment_nltk.pkl", "wb") as f:
    pickle.dump((classifier, sentim_analyzer), f)
print("\n💾 Model saved as duolingo_sentiment_nltk.pkl")

# ----------------------------
# 8️⃣ Example predictions
# ----------------------------
sample_texts = [
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

print("\n🧪 Sample predictions:")
for text in sample_texts:
    tokens = preprocess_text(text)
    tokens = mark_negation(tokens)
    features = sentim_analyzer.extract_features(tokens)
    label = classifier.classify(features)
    prob_dist = classifier.prob_classify(features)
    print(f"\n{text}")
    print(f"Predicted: {label.upper()} | Probabilities:")
    for l in prob_dist.samples():
        print(f"  {l}: {prob_dist.prob(l):.3f}")
