import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load the model package
# ----------------------------
print("📂 Loading trained classifier model package...")
with open('duolingo_sentiment_classifier.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extract components from the package
classifier = model_package['classifier']
word_features = model_package['word_features']
positive_words = model_package['positive_words']
negative_words = model_package['negative_words']

print("✅ Classifier loaded successfully!")
print(f"   - Word features: {len(word_features)}")
print(f"   - Positive words: {len(positive_words)}")
print(f"   - Negative words: {len(negative_words)}")

# ----------------------------
# Load dataset
# ----------------------------
print("\n📊 Loading dataset...")
df = pd.read_csv("tweets_duolingo_preprocessed_nltk.csv")
print(f"Total samples: {len(df)}")

# Remove rows with missing clean_text_joined
df_clean = df.dropna(subset=['clean_text_joined'])
print(f"Valid samples: {len(df_clean)} (removed {len(df) - len(df_clean)} rows with missing data)")


# ----------------------------
# Feature Extraction Function (using packaged word_features)
# ----------------------------
def extract_features(tokens, word_features, positive_words, negative_words):
    """Extract features from tokenized text using packaged components"""
    if isinstance(tokens, str):
        tokens = tokens.split()

    features = {}

    # Word features (presence of top words)
    token_set = set(tokens)
    for word in word_features:
        features[f'contains({word})'] = (word in token_set)

    # Additional features
    features['word_count'] = len(tokens)
    features['avg_word_length'] = sum(len(w) for w in tokens) / len(tokens) if tokens else 0

    # Sentiment indicators
    features['has_positive_words'] = any(word in positive_words for word in tokens)
    features['has_negative_words'] = any(word in negative_words for word in tokens)
    features['positive_word_count'] = sum(1 for word in tokens if word in positive_words)
    features['negative_word_count'] = sum(1 for word in tokens if word in negative_words)

    # Out-of-vocabulary feature (words not in training features)
    oov_count = sum(1 for word in tokens if word not in word_features)
    features['oov_ratio'] = oov_count / len(tokens) if tokens else 0

    return features


# ----------------------------
# Apply classifier to entire dataset
# ----------------------------
print("\n🤖 Applying classifier to dataset...")

predictions = []
confidence_scores = []
positive_probs = []
negative_probs = []
neutral_probs = []

for idx, row in df.iterrows():
    # Get tokens
    tokens = row['clean_text_joined'].split() if isinstance(row['clean_text_joined'], str) else []

    # Extract features using packaged word_features
    features = extract_features(tokens, word_features, positive_words, negative_words)

    # Predict
    prediction = classifier.classify(features)

    # Get probability distribution
    prob_dist = classifier.prob_classify(features)
    confidence = prob_dist.prob(prediction)

    predictions.append(prediction)
    confidence_scores.append(confidence)

    # Store probabilities for each class
    positive_probs.append(prob_dist.prob('positive'))
    negative_probs.append(prob_dist.prob('negative'))
    neutral_probs.append(prob_dist.prob('neutral'))

    # Progress indicator
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(df)} samples...")

# Add predictions to dataframe
df['sentiment'] = predictions
df['confidence_score'] = confidence_scores
df['prob_positive'] = positive_probs
df['prob_negative'] = negative_probs
df['prob_neutral'] = neutral_probs

print(f"\n✅ Classification complete!")

# ----------------------------
# Analysis & Evaluation
# ----------------------------
print("\n" + "=" * 60)
print("📊 SENTIMENT ANALYSIS RESULTS")
print("=" * 60)

# 1. Overall sentiment distribution
print("\n1️⃣ Predicted Sentiment Distribution:")
print(df['sentiment'].value_counts())
print("\nPercentage distribution:")
sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
for sentiment, pct in sentiment_pct.items():
    print(f"  {sentiment}: {pct:.2f}%")

# 2. Compare with actual labels (if available)
if 'label' in df.columns:
    print("\n2️⃣ Actual vs Predicted Comparison:")
    print(f"\nActual label distribution:")
    print(df['label'].value_counts())

    # Accuracy
    matches = (df['label'] == df['sentiment']).sum()
    total = len(df)
    accuracy = matches / total
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Confusion Matrix
    print("\n3️⃣ Confusion Matrix:")
    confusion_df = pd.crosstab(df['label'], df['sentiment'],
                               rownames=['Actual'], colnames=['Predicted'])
    print(confusion_df)

    # Per-class metrics
    print("\n4️⃣ Per-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for sentiment in ['positive', 'negative', 'neutral']:
        tp = len(df[(df['label'] == sentiment) & (df['sentiment'] == sentiment)])
        fp = len(df[(df['label'] != sentiment) & (df['sentiment'] == sentiment)])
        fn = len(df[(df['label'] == sentiment) & (df['sentiment'] != sentiment)])
        support = len(df[df['label'] == sentiment])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{sentiment:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

# 5. Confidence analysis
print("\n5️⃣ Confidence Score Analysis:")
print(f"Average confidence: {df['confidence_score'].mean():.4f}")
print(f"Median confidence: {df['confidence_score'].median():.4f}")
print(f"Min confidence: {df['confidence_score'].min():.4f}")
print(f"Max confidence: {df['confidence_score'].max():.4f}")

# Confidence distribution by sentiment
print("\nAverage confidence by sentiment:")
for sentiment in ['positive', 'negative', 'neutral']:
    avg_conf = df[df['sentiment'] == sentiment]['confidence_score'].mean()
    print(f"  {sentiment}: {avg_conf:.4f}")

# Low confidence predictions
low_confidence = df[df['confidence_score'] < 0.5]
print(f"\nLow confidence predictions (< 0.5): {len(low_confidence)} ({len(low_confidence) / len(df) * 100:.2f}%)")

# ----------------------------
# Sample predictions by sentiment
# ----------------------------
print("\n6️⃣ Sample Predictions by Sentiment:")

for sentiment in ['positive', 'negative', 'neutral']:
    print(f"\n{'=' * 60}")
    print(f"--- {sentiment.upper()} Examples (Top 3 by confidence) ---")
    print('=' * 60)

    samples = df[df['sentiment'] == sentiment].nlargest(3, 'confidence_score')

    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{i}. Text: {row['text'][:150]}...")
        print(f"   Confidence: {row['confidence_score']:.3f}")
        print(
            f"   Probabilities: Pos={row['prob_positive']:.3f}, Neg={row['prob_negative']:.3f}, Neu={row['prob_neutral']:.3f}")
        if 'label' in df.columns:
            correct = "✓" if row['label'] == row['sentiment'] else "✗"
            print(f"   Actual label: {row['label']} {correct}")

# ----------------------------
# Misclassification analysis
# ----------------------------
if 'label' in df.columns:
    print("\n7️⃣ Misclassification Analysis:")
    misclassified = df[df['label'] != df['sentiment']]
    print(f"\nTotal misclassified: {len(misclassified)} ({len(misclassified) / len(df) * 100:.2f}%)")

    # Misclassification patterns
    print("\nMisclassification patterns:")
    for actual in ['positive', 'negative', 'neutral']:
        for predicted in ['positive', 'negative', 'neutral']:
            if actual != predicted:
                count = len(df[(df['label'] == actual) & (df['sentiment'] == predicted)])
                if count > 0:
                    print(f"  {actual} → {predicted}: {count}")

    print("\n🔍 Sample misclassifications (lowest confidence):")
    misclassified_sorted = misclassified.nsmallest(5, 'confidence_score')
    for i, (idx, row) in enumerate(misclassified_sorted.iterrows(), 1):
        print(f"\n{i}. Text: {row['text'][:150]}...")
        print(f"   Actual: {row['label']} | Predicted: {row['sentiment']}")
        print(f"   Confidence: {row['confidence_score']:.3f}")
        print(
            f"   Probabilities: Pos={row['prob_positive']:.3f}, Neg={row['prob_negative']:.3f}, Neu={row['prob_neutral']:.3f}")

# ----------------------------
# Save results
# ----------------------------
print("\n" + "=" * 60)
print("💾 Saving results...")
print("=" * 60)

# Save full dataset with predictions
df.to_csv("tweets_duolingo_sentiment_classifier.csv", index=False, encoding='utf-8-sig')
print("✅ Full dataset saved as 'tweets_duolingo_sentiment_classifier.csv'")

# Save summary statistics
summary = {
    'total_samples': len(df),
    'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
    'average_confidence': float(df['confidence_score'].mean()),
    'median_confidence': float(df['confidence_score'].median()),
    'low_confidence_count': len(low_confidence)
}

if 'label' in df.columns:
    summary['accuracy'] = float(accuracy)
    summary['misclassified_count'] = len(misclassified)
    summary['misclassification_rate'] = float(len(misclassified) / len(df))

import json

with open('sentiment_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Summary saved as 'sentiment_analysis_summary.json'")
#
# # ----------------------------
# # Create visualizations
# # ----------------------------
# print("\n📊 Creating visualizations...")
#
# try:
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#
#     # 1. Sentiment distribution
#     sentiment_counts = df['sentiment'].value_counts()
#     colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
#     bar_colors = [colors.get(s, 'blue') for s in sentiment_counts.index]
#     axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
#     axes[0, 0].set_title('Predicted Sentiment Distribution', fontsize=14, fontweight='bold')
#     axes[0, 0].set_xlabel('Sentiment')
#     axes[0, 0].set_ylabel('Count')
#     for i, v in enumerate(sentiment_counts.values):
#         axes[0, 0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
#
#     # 2. Confidence distribution
#     axes[0, 1].hist(df['confidence_score'], bins=30, color='skyblue', edgecolor='black')
#     axes[0, 1].set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
#     axes[0, 1].set_xlabel('Confidence Score')
#     axes[0, 1].set_ylabel('Frequency')
#     axes[0, 1].axvline(df['confidence_score'].mean(), color='red', linestyle='--',
#                        label=f'Mean: {df["confidence_score"].mean():.3f}', linewidth=2)
#     axes[0, 1].legend()
#
#     # 3. Probability distribution by sentiment
#     df.boxplot(column=['prob_positive', 'prob_negative', 'prob_neutral'], ax=axes[0, 2])
#     axes[0, 2].set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
#     axes[0, 2].set_ylabel('Probability')
#     axes[0, 2].set_xticklabels(['Positive', 'Negative', 'Neutral'], rotation=45)
#
#     # 4. Confusion matrix (if labels available)
#     if 'label' in df.columns:
#         confusion_matrix = pd.crosstab(df['label'], df['sentiment'])
#         sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
#                     cbar_kws={'label': 'Count'})
#         axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
#         axes[1, 0].set_xlabel('Predicted')
#         axes[1, 0].set_ylabel('Actual')
#     else:
#         axes[1, 0].text(0.5, 0.5, 'No actual labels available',
#                         ha='center', va='center', fontsize=12)
#         axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
#
#     # 5. Confidence by sentiment
#     df.boxplot(column='confidence_score', by='sentiment', ax=axes[1, 1])
#     axes[1, 1].set_title('Confidence Score by Predicted Sentiment', fontsize=14, fontweight='bold')
#     axes[1, 1].set_xlabel('Predicted Sentiment')
#     axes[1, 1].set_ylabel('Confidence Score')
#     plt.sca(axes[1, 1])
#     plt.xticks(rotation=0)
#
#     # 6. Sentiment proportions (pie chart)
#     axes[1, 2].pie(sentiment_counts.values, labels=sentiment_counts.index,
#                    autopct='%1.1f%%', colors=bar_colors, startangle=90)
#     axes[1, 2].set_title('Sentiment Proportions', fontsize=14, fontweight='bold')
#
#     plt.tight_layout()
#     plt.savefig('sentiment_analysis_visualizations.png', dpi=300, bbox_inches='tight')
#     print("✅ Visualizations saved as 'sentiment_analysis_visualizations.png'")
#     plt.close()
#
# except Exception as e:
#     print(f"⚠️ Could not create visualizations: {e}")
#
# print("\n" + "=" * 60)
# print("✅ SENTIMENT ANALYSIS COMPLETE!")
# print("=" * 60)
# print("\nGenerated files:")
# print("  1. tweets_duolingo_with_predictions.csv - Full dataset with predictions")
# print("  2. sentiment_analysis_summary.json - Summary statistics")
# print("  3. sentiment_analysis_visualizations.png - Visualization charts")
# print("\nThe CSV includes these new columns:")
# print("  • sentiment - The predicted class")
# print("  • confidence_score - Confidence of prediction (0-1)")
# print("  • prob_positive - Probability of po