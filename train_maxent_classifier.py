import pandas as pd
import nltk
import pickle
import random
import ast
from nltk.classify import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.metrics import precision, recall, f_measure, ConfusionMatrix
from collections import defaultdict

# =====================================================
# 1️⃣ Load Data
# =====================================================
df = pd.read_csv("tweets_to_train.csv")

# Convert stringified token lists back to Python lists
df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# =====================================================
# 2️⃣ Feature Extraction
# =====================================================
def extract_features(tokens):
    """Convert list of tokens → feature dictionary"""
    return {word: True for word in tokens}

def prepare_dataset(df):
    """Convert DataFrame to feature-label pairs"""
    data = [(extract_features(tokens), label) for tokens, label in zip(df['tokens'], df['label'])]
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]

# =====================================================
# 3️⃣ Train MaxEnt Classifier
# =====================================================
def train_maxent_classifier(train_set, test_set, algorithm='IIS', max_iter=25):
    print(f"\n🤖 Training NLTK MaxEnt Classifier ({algorithm})...")
    classifier = MaxentClassifier.train(
        train_set,
        algorithm=algorithm,
        max_iter=max_iter,
        trace=1
    )

    # Accuracy
    train_acc = accuracy(classifier, train_set)
    test_acc = accuracy(classifier, test_set)
    print(f"\n📈 Accuracy:\nTrain: {train_acc:.3f}\nTest:  {test_acc:.3f}")

    # Top features
    print("\n🧐 Top Informative Features:")
    classifier.show_most_informative_features(10)

    return classifier

# =====================================================
# 4️⃣ Metrics: Precision, Recall, F1, Confusion Matrix
# =====================================================
def evaluate_model(classifier, test_set):
    """Compute precision, recall, F1, and confusion matrix."""
    print("\n📊 Detailed Evaluation:")

    # Extract gold and predicted labels
    gold = [label for (_, label) in test_set]
    predictions = [classifier.classify(features) for (features, _) in test_set]

    # Build confusion matrix
    cm = ConfusionMatrix(gold, predictions)

    # Compute per-class metrics
    refsets = defaultdict(set)
    testsets = defaultdict(set)

    for i, (features, label) in enumerate(test_set):
        refsets[label].add(i)
        testsets[predictions[i]].add(i)

    labels = sorted(set(gold))

    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)

    for label in labels:
        p = precision(refsets[label], testsets[label]) or 0
        r = recall(refsets[label], testsets[label]) or 0
        f1 = f_measure(refsets[label], testsets[label]) or 0
        print(f"{label:<10} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

    print("\n🧩 Confusion Matrix:")
    print(cm)

# =====================================================
# 5️⃣ Run training and evaluation
# =====================================================
train_set, test_set = prepare_dataset(df)
print(f"Training samples: {len(train_set)}, Testing samples: {len(test_set)}")

classifier = train_maxent_classifier(train_set, test_set, algorithm='IIS', max_iter=25)

# Evaluate
evaluate_model(classifier, test_set)

# =====================================================
# 6️⃣ Example prediction + Save model
# =====================================================
def predict_sentiment(text, classifier):
    tokens = nltk.word_tokenize(text.lower())
    features = extract_features(tokens)
    return classifier.classify(features)

print("\n🗣 Example prediction:")
print(predict_sentiment("I love Duolingo but the new update is terrible", classifier))

# with open("maxent_duolingo_model_1.pkl", "wb") as f:
#     pickle.dump(classifier, f)
# print("\n💾 Model saved as maxent_duolingo_model.pkl")

# # ----------------------------
# # Detailed Classification Report
# # ----------------------------
# print("\n📊 Detailed Classification Report:")
#
# # Get predictions
# predictions = []
# actuals = []
#
# for features, label in test_set:
#     pred = classifier.classify(features)
#     predictions.append(pred)
#     actuals.append(label)
#
# # Calculate metrics per class
# from collections import defaultdict
#
# metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
#
# for actual, pred in zip(actuals, predictions):
#     for label in ['positive', 'negative', 'neutral']:
#         if actual == label and pred == label:
#             metrics[label]['tp'] += 1
#         elif actual != label and pred == label:
#             metrics[label]['fp'] += 1
#         elif actual == label and pred != label:
#             metrics[label]['fn'] += 1
#         else:
#             metrics[label]['tn'] += 1
#
# # Print metrics
# print("\n{:<12} {:<12} {:<12} {:<12}".format("Class", "Precision", "Recall", "F1-Score"))
# print("-" * 50)
#
# for label in ['positive', 'negative', 'neutral']:
#     tp = metrics[label]['tp']
#     fp = metrics[label]['fp']
#     fn = metrics[label]['fn']
#
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#
#     print("{:<12} {:<12.4f} {:<12.4f} {:<12.4f}".format(label, precision, recall, f1))
#
# # Confusion Matrix
# print("\n🎯 Confusion Matrix:")
# confusion_matrix = defaultdict(lambda: defaultdict(int))
# for actual, pred in zip(actuals, predictions):
#     confusion_matrix[actual][pred] += 1
#
# print("\n{:<12} {:<12} {:<12} {:<12}".format("", "Positive", "Negative", "Neutral"))
# print("-" * 50)
# for actual in ['positive', 'negative', 'neutral']:
#     print("{:<12} {:<12} {:<12} {:<12}".format(
#         actual.capitalize(),
#         confusion_matrix[actual]['positive'],
#         confusion_matrix[actual]['negative'],
#         confusion_matrix[actual]['neutral']
#     ))

# # ----------------------------
# # Create Model Package (Classifier + word_features)
# # ----------------------------
# print("\n💾 Saving model with word_features...")
#
# # Package everything needed for prediction
# model_package = {
#     'classifier': classifier,
#     'word_features': word_features,
#     'positive_words': {'good', 'great', 'love', 'excellent', 'amazing', 'best', 'fun',
#                        'happy', 'enjoy', 'helpful', 'thank', 'perfect', 'awesome', 'nice',
#                        'wonderful', 'fantastic', 'brilliant', 'cool', 'glad'},
#     'negative_words': {'bad', 'hate', 'terrible', 'worst', 'annoying', 'frustrating',
#                        'useless', 'disappointed', 'angry', 'lose', 'broke', 'sucks',
#                        'awful', 'horrible', 'stupid', 'shit', 'fuck', 'damn'}
# }
#
# with open('duolingo_sentiment_classifier.pkl', 'wb') as f:
#     pickle.dump(model_package, f)
#
# print("✅ Model package saved as 'duolingo_sentiment_classifier.pkl'")
# print("   Package includes: classifier, word_features, sentiment_words")
#
# # ----------------------------
# # Test with sample predictions
# # ----------------------------
# print("\n🧪 Sample Predictions:")
#
# test_texts = [
#     "I love Duolingo! It's so fun and helpful for learning languages.",
#     "Lost my streak and the energy system ruins the app.",
#     "Just completed my daily lesson on Duolingo.",
#     "The new update is annoying and frustrating.",
#     "Duolingo helped me learn Spanish quickly. Best app ever!",
#     "Come on Duolingo How can this man confidently peruse the glamorous Parisian shopping districts without your immediate help",
#     "Is that why my Duolingo stopped working",
#     "Duolingo episode would have made me want to kill myself if this was a finale to a show I was genuinely invested in and was not watching for the LOLs That is all I will say about it",
#     "Im glad i know what this means my Duolingo lessons paying off",
#     "Amazon is cloud services unit Amazon Web Services AWS has been hit by an outage causing connectivity issues for many companies around the world The services disrupted include several popular websites and apps including Fortnite Snapchat and Duolingo",
#     "My Duolingo streak better not die I m scared of that bird",
#     "Duolingo actually helped me speak and read espa ol a lot better",
#     "I study italian on duolingo and it full of and diversity stuff Just unbelievable",
#     "for your information Snapchat Fortnite Clash Royale Roblox Alexa Amazon Amazon Prime Video all Amazon devices Duolingo Canva The NY Times Apple TV PUBG AND me are all down rn lol",
#     "Duolingo is gracious and calling it a maintenance break"
# ]
#
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
#
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
#
# def preprocess_text(text):
#     """Preprocess new text for prediction"""
#     # Clean
#     text = text.lower()
#     tokens = word_tokenize(text)
#     # Remove stopwords and lemmatize
#     tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
#     return tokens
#
#
# for text in test_texts:
#     tokens = preprocess_text(text)
#     features = extract_features(tokens, word_features)
#     prediction = classifier.classify(features)
#     prob_dist = classifier.prob_classify(features)
#     confidence = prob_dist.prob(prediction)
#
#     print(f"\nText: {text}")
#     print(f"Prediction: {prediction.upper()} (confidence: {confidence:.3f})")
#
# print("\n" + "=" * 60)
# print("✅ TRAINING COMPLETE!")
# print("=" * 60)
# print("\nModel package contains:")
# print(f"  • Trained classifier")
# print(f"  • {len(word_features)} word features")
# print(f"  • Sentiment word dictionaries")
# print("\nYou can now use this model in other scripts without redefining features!")