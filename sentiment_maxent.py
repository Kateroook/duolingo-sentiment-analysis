import pandas as pd
import ast
import pickle
from nltk.classify import MaxentClassifier

# ==========================================================
# 1️⃣ Load trained model
# ==========================================================
MODEL_PATH = "maxent_duolingo_model_1.pkl"

print("🔹 Loading trained MaxEnt classifier...")
with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)
print("✅ Model loaded successfully.")

# ==========================================================
# 2️⃣ Load dataset
# ==========================================================
INPUT_FILE = "tweets_duolingo_preprocessed.csv"
OUTPUT_FILE = "tweets_duolingo_sentiment_maxent.csv"

print(f"\n📂 Loading data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} tweets")

# ==========================================================
# 3️⃣ Prepare tokens
# ==========================================================
# 'tokens' column looks like: "['good', 'way', 'duolingo', 'scam']"
# Convert string representation back to Python list safely
def parse_tokens(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    elif isinstance(value, list):
        return value
    else:
        return []

df['tokens'] = df['tokens'].apply(parse_tokens)

# ==========================================================
# 4️⃣ Define feature extractor
# ==========================================================
def extract_features(tokens):
    """Convert token list into a {word: True} feature dict."""
    return {word: True for word in tokens if isinstance(word, str)}

# ==========================================================
# 5️⃣ Predict sentiment for each tweet
# ==========================================================
print("\n🤖 Analyzing tweets...")

predictions = []
probabilities = []

for tokens in df['tokens']:
    feats = extract_features(tokens)
    if not feats:
        predictions.append("neutral")
        probabilities.append({'neutral': 1.0, 'positive': 0.0, 'negative': 0.0})
        continue

    label = classifier.classify(feats)
    prob_dist = classifier.prob_classify(feats)

    predictions.append(label)
    probabilities.append({
        'positive': round(prob_dist.prob('positive'), 3) if 'positive' in prob_dist.samples() else 0.0,
        'negative': round(prob_dist.prob('negative'), 3) if 'negative' in prob_dist.samples() else 0.0,
        'neutral':  round(prob_dist.prob('neutral'), 3) if 'neutral' in prob_dist.samples() else 0.0
    })

# ==========================================================
# 6️⃣ Add predictions to dataframe
# ==========================================================
df['predicted_label'] = predictions
df['prob_positive'] = [p['positive'] for p in probabilities]
df['prob_negative'] = [p['negative'] for p in probabilities]
df['prob_neutral'] = [p['neutral'] for p in probabilities]

# ==========================================================
# 7️⃣ Save output
# ==========================================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Results saved to: {OUTPUT_FILE}")

# ==========================================================
# 8️⃣ Quick summary
# ==========================================================
print("\n📊 Sentiment distribution:")
print(df['predicted_label'].value_counts())

print("\n✅ Analysis complete!")
