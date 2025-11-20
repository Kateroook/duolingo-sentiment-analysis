
import pandas as pd
import pickle
from nltk.metrics import precision, recall, f_measure, ConfusionMatrix
from collections import defaultdict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# ----------------------------
# Load preprocessed labeled data
# ----------------------------
df = pd.read_csv("tweets_to_train.csv")

# Convert stringified lists to actual Python lists
import ast

df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# ----------------------------
# Load trained MaxEnt classifier
# ----------------------------
with open("maxent_duolingo_model_.pkl", "rb") as f:
    maxent_classifier = pickle.load(f)


# Feature extraction function
def extract_features(tokens):
    return {word: True for word in tokens}


# ----------------------------
# VADER setup
# ----------------------------
sid = SentimentIntensityAnalyzer()


def vader_label(text, pos_thresh=0.05, neg_thresh=-0.05):
    score = sid.polarity_scores(str(text))['compound']
    if score >= pos_thresh:
        return 'positive'
    elif score <= neg_thresh:
        return 'negative'
    else:
        return 'neutral'


# ----------------------------
# TextBlob setup
# ----------------------------
def textblob_label(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.05:
        return 'positive'
    elif polarity < -0.05:
        return 'negative'
    else:
        return 'neutral'


# ----------------------------
# Predict labels
# ----------------------------
methods = {
    'MaxEnt': [],
    'VADER': [],
    'TextBlob': []
}

gold = list(df['label'])

for tokens, text in zip(df['tokens'], df['clean_text_joined']):
    # MaxEnt prediction
    features = extract_features(tokens)
    pred_maxent = maxent_classifier.classify(features)
    methods['MaxEnt'].append(pred_maxent)

    # VADER prediction
    pred_vader = vader_label(text)
    methods['VADER'].append(pred_vader)

    # TextBlob prediction
    pred_tb = textblob_label(text)
    methods['TextBlob'].append(pred_tb)


# ----------------------------
# Evaluation function
# ----------------------------
def evaluate(predictions, gold_labels, method_name):
    print(f"\n===== {method_name} Evaluation =====")

    # Confusion Matrix
    cm = ConfusionMatrix(gold_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Precision, Recall, F1 per class
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    for i, l in enumerate(gold_labels):
        refsets[l].add(i)
        testsets[predictions[i]].add(i)

    labels = sorted(set(gold_labels))
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    for label in labels:
        p = precision(refsets[label], testsets[label]) or 0
        r = recall(refsets[label], testsets[label]) or 0
        f1 = f_measure(refsets[label], testsets[label]) or 0
        print(f"{label:<10} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")


# ----------------------------
# Run evaluation for each method
# ----------------------------
for method, preds in methods.items():
    evaluate(preds, gold, method)
