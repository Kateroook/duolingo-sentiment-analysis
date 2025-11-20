import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
from itertools import chain
from ast import literal_eval
import numpy as np
from datetime import datetime, timedelta

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

# Define consistent color scheme
SENTIMENT_COLORS = {
    'negative': '#FF6B6B',  # Coral red
    'neutral': '#FFD93D',  # Golden yellow
    'positive': '#6BCB77'  # Light green
}

# Define key events
KEY_EVENTS = {
    '2024-10-20': 'AWS Outage',
    '2024-10-21': 'Louvre Robbery (Duo-Madonna meme)'
}


def get_sentiment_colors(sentiment_order):
    """Returns list of colors matching the sentiment order"""
    return [SENTIMENT_COLORS[s] for s in sentiment_order]


# ----------------------------
# Load data
# ----------------------------
print("=" * 60)
print("DUOLINGO SENTIMENT ANALYSIS - EVENT-FOCUSED EDA")
print("=" * 60)

df = pd.read_csv("tweets_duolingo_sentiment_maxent.csv")

# Clean data
df['clean_text_joined'] = df['clean_text_joined'].astype(str).fillna("")
df['text'] = df['text'].astype(str).fillna("")

# Parse/create date column
if 'created_at' in df.columns:
    df['date'] = pd.to_datetime(df['created_at'], errors='coerce')
elif 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    print("⚠️ No date column found. Creating dates for October 2024.")
    # Distribute tweets across October 2024
    start_date = pd.Timestamp('2024-10-01')
    end_date = pd.Timestamp('2024-10-31')
    df['date'] = pd.date_range(start=start_date, end=end_date, periods=len(df))

df['date_only'] = df['date'].dt.date
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = df['date'].dt.hour

# Use predicted_sentiment if available, otherwise use label
if 'sentiment' in df.columns:
    df['sentiment'] = df['sentiment']
elif 'label' in df.columns:
    df['sentiment'] = df['label']
else:
    print("⚠️ No sentiment column found!")

# Convert engagement metrics to numeric
for col in ['retweets', 'likes', 'replies', 'quotes']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


print(f"📊 Dataset: {len(df)} tweets")
print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"💬 Sentiment distribution:\n{df['sentiment'].value_counts()}\n")


# ----------------------------
# 1️⃣ OVERALL SENTIMENT DISTRIBUTION
# ----------------------------
print("=" * 60)
print("1️⃣ OVERALL SENTIMENT DISTRIBUTION")
print("=" * 60)

sentiment_counts = df['sentiment'].value_counts()
sentiment_pct = (sentiment_counts / len(df) * 100).round(2)

for sent, count in sentiment_counts.items():
    print(f"{sent.capitalize()}: {count} ({sentiment_pct[sent]}%)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Pie chart
sentiment_order = ['negative', 'neutral', 'positive']
pie_data = [sentiment_counts.get(s, 0) for s in sentiment_order]
pie_colors = get_sentiment_colors(sentiment_order)

axes[0].pie(pie_data, labels=[s.capitalize() for s in sentiment_order],
            autopct='%1.1f%%', startangle=140, colors=pie_colors,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title("Sentiment Distribution", fontsize=14, fontweight='bold')

# Bar chart
bar_data = df['sentiment'].copy()
bar_data = pd.Categorical(bar_data, categories=sentiment_order, ordered=True)
df_temp = pd.DataFrame({'sentiment': bar_data})
sns.countplot(data=df_temp, x='sentiment', palette=pie_colors, ax=axes[1], order=sentiment_order)
axes[1].set_title("Tweet Count per Sentiment", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Sentiment", fontweight='bold')
axes[1].set_ylabel("Count", fontweight='bold')
for container in axes[1].containers:
    axes[1].bar_label(container, fontweight='bold')

# Confidence distribution (if available)
if 'confidence_score' in df.columns:
    for sentiment in sentiment_order:
        subset = df[df['sentiment'] == sentiment]['confidence_score']
        axes[2].hist(subset, bins=20, alpha=0.6, label=sentiment.capitalize(),
                     color=SENTIMENT_COLORS[sentiment])
    axes[2].set_title("Confidence Score Distribution", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Confidence Score", fontweight='bold')
    axes[2].set_ylabel("Frequency", fontweight='bold')
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, 'No confidence data available',
                 ha='center', va='center', fontsize=12)
    axes[2].set_title("Confidence Distribution", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 01_sentiment_distribution.png\n")

# ----------------------------
# 2️⃣ EVENT-FOCUSED TIMELINE ANALYSIS
# ----------------------------
print("=" * 60)
print("2️⃣ EVENT-FOCUSED TIMELINE ANALYSIS")
print("=" * 60)

# Daily sentiment aggregation
daily_sentiment = df.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)
daily_sentiment = daily_sentiment.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)
daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100

# Total tweets per day
daily_total = df.groupby('date_only').size()

print(f"📊 Tracking {len(daily_sentiment)} days")
print(f"Average tweets/day: {daily_total.mean():.1f}")
print(f"Peak day: {daily_total.idxmax()} ({daily_total.max()} tweets)\n")

# Create comprehensive timeline visualization
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# 1. Absolute counts with events
ax1 = fig.add_subplot(gs[0, :])
line_colors = get_sentiment_colors(['negative', 'neutral', 'positive'])
for i, sentiment in enumerate(['negative', 'neutral', 'positive']):
    ax1.plot(daily_sentiment.index, daily_sentiment[sentiment],
             marker='o', label=sentiment.capitalize(),
             color=line_colors[i], linewidth=2.5, markersize=6)

# Add event markers
for event_date, event_name in KEY_EVENTS.items():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_sentiment.index:
        ax1.axvline(x=event_dt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(event_dt, ax1.get_ylim()[1] * 0.95, event_name,
                 rotation=90, va='top', ha='right', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_title("Daily Sentiment Trends with Key Events", fontsize=16, fontweight='bold')
ax1.set_xlabel("Date", fontweight='bold', fontsize=12)
ax1.set_ylabel("Number of Tweets", fontweight='bold', fontsize=12)
ax1.legend(title="Sentiment", fontsize=11, title_fontsize=12)


ax1.grid(True, alpha=0.3)

# 2. Percentage trends
ax2 = fig.add_subplot(gs[1, :])
for i, sentiment in enumerate(['negative', 'neutral', 'positive']):
    ax2.plot(daily_sentiment_pct.index, daily_sentiment_pct[sentiment],
             marker='o', label=sentiment.capitalize(),
             color=line_colors[i], linewidth=2.5, markersize=6)

for event_date, event_name in KEY_EVENTS.items():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_sentiment_pct.index:
        ax2.axvline(x=event_dt, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax2.set_title("Daily Sentiment Percentage", fontsize=16, fontweight='bold')
ax2.set_xlabel("Date", fontweight='bold', fontsize=12)
ax2.set_ylabel("Percentage (%)", fontweight='bold', fontsize=12)
ax2.legend(title="Sentiment", fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Stacked area chart
ax3 = fig.add_subplot(gs[2, :])
ax3.stackplot(daily_sentiment.index,
              daily_sentiment['negative'],
              daily_sentiment['neutral'],
              daily_sentiment['positive'],
              labels=['Negative', 'Neutral', 'Positive'],
              colors=line_colors, alpha=0.8)

for event_date, event_name in KEY_EVENTS.items():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_sentiment.index:
        ax3.axvline(x=event_dt, color='darkred', linestyle='--', linewidth=2.5, alpha=0.9)

ax3.set_title("Sentiment Distribution Over Time (Stacked)", fontsize=16, fontweight='bold')
ax3.set_xlabel("Date", fontweight='bold', fontsize=12)
ax3.set_ylabel("Number of Tweets", fontweight='bold', fontsize=12)
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Net sentiment score
ax4 = fig.add_subplot(gs[3, 0])
daily_score = daily_sentiment['positive'] - daily_sentiment['negative']
ax4.plot(daily_score.index, daily_score.values, marker='o',
         linewidth=2.5, color='#4A90E2', markersize=6)
ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=2)
ax4.fill_between(daily_score.index, daily_score.values, 0,
                 where=(daily_score.values >= 0), color=SENTIMENT_COLORS['positive'],
                 alpha=0.4, label='Positive net')
ax4.fill_between(daily_score.index, daily_score.values, 0,
                 where=(daily_score.values < 0), color=SENTIMENT_COLORS['negative'],
                 alpha=0.4, label='Negative net')

for event_date in KEY_EVENTS.keys():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_score.index:
        ax4.axvline(x=event_dt, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax4.set_title("Net Sentiment Score (Positive - Negative)", fontsize=14, fontweight='bold')
ax4.set_xlabel("Date", fontweight='bold')
ax4.set_ylabel("Net Score", fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Daily tweet volume
ax5 = fig.add_subplot(gs[3, 1])
ax5.bar(daily_total.index, daily_total.values, color='steelblue', alpha=0.7)

for event_date in KEY_EVENTS.keys():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_total.index:
        ax5.axvline(x=event_dt, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax5.set_title("Daily Tweet Volume", fontsize=14, fontweight='bold')
ax5.set_xlabel("Date", fontweight='bold')
ax5.set_ylabel("Number of Tweets", fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig('02_event_timeline_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 02_event_timeline_analysis.png\n")

# ----------------------------
# 3️⃣ EVENT IMPACT ANALYSIS
# ----------------------------
print("=" * 60)
print("3️⃣ EVENT IMPACT ANALYSIS")
print("=" * 60)

# Define event windows
event_windows = {
    'AWS Outage (Oct 20)': {
        'date': '2025-10-20',
        'before': 2,  # days before
        'after': 2  # days after
    },
    'Louvre Robbery (Oct 21)': {
        'date': '2025-10-21',
        'before': 1,
        'after': 2
    }
}

fig, axes = plt.subplots(len(event_windows), 2, figsize=(16, 6 * len(event_windows)))
if len(event_windows) == 1:
    axes = axes.reshape(1, -1)

for idx, (event_name, event_info) in enumerate(event_windows.items()):
    event_date = datetime.strptime(event_info['date'], '%Y-%m-%d').date()
    before_days = event_info['before']
    after_days = event_info['after']

    start_date = event_date - timedelta(days=before_days)
    end_date = event_date + timedelta(days=after_days)

    # Filter data for this window
    mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
    event_df = df[mask]

    if len(event_df) == 0:
        print(f"⚠️ No data for {event_name}")
        continue

    print(f"\n📌 {event_name}")
    print(f"   Window: {start_date} to {end_date}")
    print(f"   Tweets: {len(event_df)}")

    # Before vs After comparison
    before_df = df[df['date_only'] < event_date]
    after_df = df[df['date_only'] >= event_date]

    before_sent = before_df['sentiment'].value_counts(normalize=True) * 100
    after_sent = after_df['sentiment'].value_counts(normalize=True) * 100

    print(f"   Before event sentiment:")
    for sent in ['positive', 'negative', 'neutral']:
        print(f"     {sent}: {before_sent.get(sent, 0):.1f}%")
    print(f"   After event sentiment:")
    for sent in ['positive', 'negative', 'neutral']:
        print(f"     {sent}: {after_sent.get(sent, 0):.1f}%")

    # Plot 1: Sentiment trend around event
    event_daily = event_df.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)
    event_daily = event_daily.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)

    for i, sentiment in enumerate(['negative', 'neutral', 'positive']):
        axes[idx, 0].plot(event_daily.index, event_daily[sentiment],
                          marker='o', label=sentiment.capitalize(),
                          color=line_colors[i], linewidth=3, markersize=8)

    axes[idx, 0].axvline(x=event_date, color='red', linestyle='--',
                         linewidth=3, label='Event Date', alpha=0.8)
    axes[idx, 0].set_title(f"{event_name} - Sentiment Trend",
                           fontsize=14, fontweight='bold')
    axes[idx, 0].set_xlabel("Date", fontweight='bold')
    axes[idx, 0].set_ylabel("Number of Tweets", fontweight='bold')
    axes[idx, 0].legend(fontsize=10)
    axes[idx, 0].grid(True, alpha=0.3)

    # Plot 2: Before vs After comparison
    comparison_data = pd.DataFrame({
        'Before': [before_sent.get(s, 0) for s in ['negative', 'neutral', 'positive']],
        'After': [after_sent.get(s, 0) for s in ['negative', 'neutral', 'positive']]
    }, index=['Negative', 'Neutral', 'Positive'])

    comparison_data.plot(kind='bar', ax=axes[idx, 1],
                         color=['#95A5A6', '#E74C3C'], width=0.7)
    axes[idx, 1].set_title(f"{event_name} - Before vs After",
                           fontsize=14, fontweight='bold')
    axes[idx, 1].set_ylabel("Percentage (%)", fontweight='bold')
    axes[idx, 1].set_xlabel("Sentiment", fontweight='bold')
    axes[idx, 1].legend(fontsize=10)
    axes[idx, 1].set_xticklabels(axes[idx, 1].get_xticklabels(), rotation=0)
    axes[idx, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_event_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✅ Saved: 03_event_impact_analysis.png\n")

# ----------------------------
# 4️⃣ TOPIC ANALYSIS
# ----------------------------
print("=" * 60)
print("4️⃣ TOPIC ANALYSIS")
print("=" * 60)

# Enhanced topic keywords
topics = {
    'aws_outage': ['outage', 'down', 'aws', 'amazon', 'maintenance', 'crash', 'broken', 'server', 'offline', 'back', 'interruption', 'disrupt', 'provider'],
    'louvre_meme': ['louvre', 'madonna', 'mona', 'lisa', 'theft', 'robbery', 'museum', 'painting'],
    'anime': ['anime', 'episode', 'game'],
    'energy_system': ['energy', 'heart', 'hearts', 'battery', 'power', 'recharge', 'charge','stamina', 'update'],
    'streak': ['streak', 'freeze', 'freezing', 'lose', 'lost', 'day', 'maintain', 'flame', 'xp', 'chest','league'],
    'learning': ['learn', 'learning' ,'language', 'teach', 'taught' ,'lesson', 'exercise', 'sentence', 'phrase','study', 'practice', 'course', 'profile', 'unit', 'vocab','vocabulary','grammar'],
    'complaint': ['fix', 'bug', 'trouble', 'issue', 'problem', 'terrible', 'bad', 'hate', 'irrelevant', 'annoying','frustrate', 'suck','piss', 'pathetic', 'worst', 'useless', 'scam', 'ass', 'as', 'annoy','tired','tire','cringe', 'delete', 'sad', 'disappoint', 'strange','crap'],
    'achievement': ['complete', 'challenge', 'win', 'achievement', 'milestone', 'tournament', 'score', 'stats', 'league', 'level', 'progress','point','quest'],
    'positive_feedback': ['love', 'adore','enjoy','great', 'good', 'thank','thanks', 'awesome', 'amazing', 'amaze', 'best', 'fun','cool', 'helpful','effective', 'fire', 'hero']
}


def assign_topic(text):
    text = text.lower()
    found_topics = []
    for topic, keywords in topics.items():
        if any(word in text for word in keywords):
            found_topics.append(topic)
    return found_topics if found_topics else ['other']


# Assign multiple topics
df['topics_list'] = df['clean_text_joined'].apply(assign_topic)
df['primary_topic'] = df['topics_list'].apply(lambda x: x[0])
#
# # Відфільтрувати рядки з темою 'other'
# other_df = df[df['primary_topic'] == 'other']
#
# # Вибрати лише колонки text та clean_text_joined
# other_texts = other_df[['text', 'clean_text_joined']]
#
# # Зберегти у CSV
# other_texts.to_csv('other_texts.csv', index=False, encoding='utf-8')

# Topic distribution

topic_counts = df['primary_topic'].value_counts()
print("\n📊 Topic Distribution:")
print(topic_counts)

# Create topic visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Topic distribution
axes[0, 0].barh(topic_counts.index, topic_counts.values, color='steelblue', alpha=0.8)
axes[0, 0].set_title("Tweet Distribution by Topic", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Number of Tweets", fontweight='bold')
axes[0, 0].set_ylabel("Topic", fontweight='bold')
for i, v in enumerate(topic_counts.values):
    axes[0, 0].text(v + 1, i, str(v), va='center', fontweight='bold')

# 2. Sentiment by topic (stacked percentage)
topic_sentiment = pd.crosstab(df['primary_topic'], df['sentiment'], normalize='index') * 100
topic_sentiment = topic_sentiment.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)

topic_sentiment.plot(kind='barh', stacked=True, ax=axes[0, 1],
                     color=get_sentiment_colors(['negative', 'neutral', 'positive']))
axes[0, 1].set_title("Sentiment Distribution per Topic (%)", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Percentage (%)", fontweight='bold')
axes[0, 1].set_ylabel("Topic", fontweight='bold')
axes[0, 1].legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. Topic sentiment heatmap
topic_sentiment_counts = pd.crosstab(df['primary_topic'], df['sentiment'])
topic_sentiment_counts = topic_sentiment_counts.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)

sns.heatmap(topic_sentiment_counts, annot=True, fmt='d', cmap='YlOrRd',
            ax=axes[1, 0], cbar_kws={'label': 'Count'})
axes[1, 0].set_title("Topic vs Sentiment Heatmap", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Sentiment", fontweight='bold')
axes[1, 0].set_ylabel("Topic", fontweight='bold')

# 4. Topic trends over time (top 5)
topic_counts = topic_counts[topic_counts.index != 'other']
daily_topics = df.groupby(['date_only', 'primary_topic']).size().unstack(fill_value=0)
top_5_topics = topic_counts.head(5).index

for topic in top_5_topics:
    if topic in daily_topics.columns:
        axes[1, 1].plot(daily_topics.index, daily_topics[topic],
                        marker='o', label=topic.replace('_', ' ').title(), linewidth=2)

for event_date in KEY_EVENTS.keys():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_topics.index:
        axes[1, 1].axvline(x=event_dt, color='red', linestyle='--', linewidth=2, alpha=0.6)

axes[1, 1].set_title("Topic Trends Over Time (Top 5)", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Date", fontweight='bold')
axes[1, 1].set_ylabel("Number of Tweets", fontweight='bold')
axes[1, 1].legend(fontsize=9, loc='best')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_topic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 04_topic_analysis.png\n")

# Print topic-sentiment insights
print("\n📊 Topic-Sentiment Insights:")
for topic in topic_counts.head(5).index:
    topic_data = df[df['primary_topic'] == topic]
    sent_dist = topic_data['sentiment'].value_counts()
    dominant_sentiment = sent_dist.idxmax()
    print(f"\n{topic.replace('_', ' ').title()}:")
    print(f"  Dominant sentiment: {dominant_sentiment.upper()}")
    print(f"  Distribution: {sent_dist.to_dict()}")

# ----------------------------
# 5️⃣ WORD CLOUD ANALYSIS
# ----------------------------
print("\n" + "=" * 60)
print("5️⃣ WORD CLOUD ANALYSIS BY SENTIMENT")
print("=" * 60)


def create_wordcloud_for_sentiment(df, sentiment_label):
    """Create word cloud for specific sentiment"""
    subset = df[df['sentiment'] == sentiment_label]

    if 'tokens' in subset.columns:
        # Try to parse tokens if they're strings
        subset = subset.copy()
        subset['tokens_parsed'] = subset['tokens'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) else x
        )
        all_tokens = [token for tokens in subset['tokens_parsed'] for token in tokens if isinstance(tokens, list)]
    else:
        # Use clean_text_joined
        all_tokens = ' '.join(subset['clean_text_joined']).split()

    if not all_tokens:
        print(f"⚠️ No tokens for {sentiment_label}")
        return None, None

    # Word frequency
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(20)

    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', relative_scaling=0.5,
                          max_words=100).generate(' '.join(all_tokens))

    return wordcloud, top_words


# Create word clouds for each sentiment
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for idx, sentiment in enumerate(['positive', 'negative', 'neutral']):
    print(f"\nGenerating word cloud for {sentiment.upper()}...")

    wordcloud, top_words = create_wordcloud_for_sentiment(df, sentiment)

    if wordcloud is None:
        continue

    # Word cloud
    axes[idx, 0].imshow(wordcloud, interpolation='bilinear')
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title(f"Word Cloud - {sentiment.capitalize()} Sentiment",
                           fontsize=14, fontweight='bold',
                           color=SENTIMENT_COLORS[sentiment])

    # Top words bar chart
    if top_words:
        words, freqs = zip(*top_words)
        axes[idx, 1].barh(list(words)[::-1], list(freqs)[::-1],
                          color=SENTIMENT_COLORS[sentiment], alpha=0.8)
        axes[idx, 1].set_title(f"Top 20 Words - {sentiment.capitalize()} Sentiment",
                               fontsize=14, fontweight='bold',
                               color=SENTIMENT_COLORS[sentiment])
        axes[idx, 1].set_xlabel("Frequency", fontweight='bold')
        axes[idx, 1].set_ylabel("Word", fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('05_wordclouds_by_sentiment.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 05_wordclouds_by_sentiment.png\n")

# ----------------------------
# 6️⃣ N-GRAM ANALYSIS
# ----------------------------
print("=" * 60)
print("6️⃣ N-GRAM ANALYSIS")
print("=" * 60)

# Get all tokens
if 'tokens' in df.columns:
    all_tokens = []
    for tokens in df['tokens']:
        try:
            if isinstance(tokens, str):
                tokens = literal_eval(tokens)
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
        except:
            continue
else:
    all_tokens = ' '.join(df['clean_text_joined']).split()

# Bigrams and Trigrams
bigrams_list = list(ngrams(all_tokens, 2))
trigrams_list = list(ngrams(all_tokens, 3))

bigram_freq = Counter(bigrams_list)
trigram_freq = Counter(trigrams_list)

top_bigrams = bigram_freq.most_common(15)
top_trigrams = trigram_freq.most_common(15)

print(f"\nTop 10 Bigrams:")
for bigram, count in top_bigrams[:10]:
    print(f"  {' '.join(bigram)}: {count}")

print(f"\nTop 10 Trigrams:")
for trigram, count in top_trigrams[:10]:
    print(f"  {' '.join(trigram)}: {count}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bigrams
bigram_labels = [' '.join(b) for b, _ in top_bigrams]
bigram_counts = [c for _, c in top_bigrams]
axes[0].barh(bigram_labels[::-1], bigram_counts[::-1], color='teal', alpha=0.8)
axes[0].set_title("Top 15 Bigrams", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Frequency", fontweight='bold')
axes[0].set_ylabel("Bigram", fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Trigrams
trigram_labels = [' '.join(t) for t, _ in top_trigrams]
trigram_counts = [c for _, c in top_trigrams]
axes[1].barh(trigram_labels[::-1], trigram_counts[::-1], color='coral', alpha=0.8)
axes[1].set_title("Top 15 Trigrams", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Frequency", fontweight='bold')
axes[1].set_ylabel("Trigram", fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('06_ngram_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 06_ngram_analysis.png\n")

# ----------------------------
# 7️⃣ TEMPORAL PATTERNS
# ----------------------------
print("=" * 60)
print("7️⃣ TEMPORAL PATTERNS ANALYSIS")
print("=" * 60)

# Day of week analysis
day_order = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

dow_sentiment = pd.crosstab(df['day_of_week'], df['sentiment'], normalize='index') * 100
dow_sentiment = dow_sentiment.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)
dow_counts = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)

print("\n📅 Day of Week Analysis:")
print(dow_counts)

# Hour of day analysis (if available)
hour_counts = df.groupby('hour').size()
hour_sentiment = pd.crosstab(df['hour'], df['sentiment'], normalize='index') * 100
hour_sentiment = hour_sentiment.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Day of week - tweet volume
axes[0, 0].bar(dow_counts.index, dow_counts.values, color='steelblue', alpha=0.8)
axes[0, 0].set_title("Tweet Volume by Day of Week", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Day", fontweight='bold')
axes[0, 0].set_ylabel("Number of Tweets", fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Day of week - sentiment distribution
dow_sentiment.plot(kind='bar', stacked=True, ax=axes[0, 1],
                   color=get_sentiment_colors(['negative', 'neutral', 'positive']))
axes[0, 1].set_title("Sentiment Distribution by Day of Week", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Day", fontweight='bold')
axes[0, 1].set_ylabel("Percentage (%)", fontweight='bold')
axes[0, 1].legend(title="Sentiment")
axes[0, 1].tick_params(axis='x', rotation=45)

# Hour of day - tweet volume
axes[1, 0].plot(hour_counts.index, hour_counts.values, marker='o',
                linewidth=2.5, color='darkblue', markersize=6)
axes[1, 0].set_title("Tweet Volume by Hour of Day", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Hour", fontweight='bold')
axes[1, 0].set_ylabel("Number of Tweets", fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(0, 24, 2))

# Hour of day - sentiment heatmap
hour_sentiment_pivot = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
hour_sentiment_pivot = hour_sentiment_pivot.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)
sns.heatmap(hour_sentiment_pivot.T, cmap='YlOrRd', annot=True, fmt='d',
            ax=axes[1, 1], cbar_kws={'label': 'Count'})
axes[1, 1].set_title("Sentiment by Hour Heatmap", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Hour", fontweight='bold')
axes[1, 1].set_ylabel("Sentiment", fontweight='bold')

plt.tight_layout()
plt.savefig('07_temporal_patterns.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 07_temporal_patterns.png\n")

# ----------------------------
# 8️⃣ TWEET LENGTH ANALYSIS
# ----------------------------
print("=" * 60)
print("8️⃣ TWEET LENGTH ANALYSIS")
print("=" * 60)

df['char_count'] = df['clean_text_joined'].apply(len)
df['word_count'] = df['clean_text_joined'].apply(lambda x: len(x.split()))

print(f"\n📏 Length Statistics:")
print(f"Average characters: {df['char_count'].mean():.1f}")
print(f"Median characters: {df['char_count'].median():.1f}")
print(f"Average words: {df['word_count'].mean():.1f}")
print(f"Median words: {df['word_count'].median():.1f}")

# Length by sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    subset = df[df['sentiment'] == sentiment]
    print(f"\n{sentiment.capitalize()} tweets:")
    print(f"  Avg chars: {subset['char_count'].mean():.1f}")
    print(f"  Avg words: {subset['word_count'].mean():.1f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Character count distribution
sns.histplot(data=df, x='char_count', bins=50, kde=True,
             color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title("Tweet Length Distribution (Characters)", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Number of Characters", fontweight='bold')
axes[0, 0].set_ylabel("Frequency", fontweight='bold')
axes[0, 0].axvline(df['char_count'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {df["char_count"].mean():.0f}')
axes[0, 0].legend()

# Word count distribution
sns.histplot(data=df, x='word_count', bins=50, kde=True,
             color='lightcoral', ax=axes[0, 1])
axes[0, 1].set_title("Tweet Length Distribution (Words)", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Number of Words", fontweight='bold')
axes[0, 1].set_ylabel("Frequency", fontweight='bold')
axes[0, 1].axvline(df['word_count'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {df["word_count"].mean():.0f}')
axes[0, 1].legend()

# Length by sentiment - boxplot
sentiment_order_box = ['negative', 'neutral', 'positive']
sns.boxplot(data=df, x='sentiment', y='char_count',
            palette=get_sentiment_colors(sentiment_order_box),
            order=sentiment_order_box, ax=axes[1, 0])
axes[1, 0].set_title("Character Count by Sentiment", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Sentiment", fontweight='bold')
axes[1, 0].set_ylabel("Character Count", fontweight='bold')

# Length by sentiment - violin plot
sns.violinplot(data=df, x='sentiment', y='word_count',
               palette=get_sentiment_colors(sentiment_order_box),
               order=sentiment_order_box, ax=axes[1, 1])
axes[1, 1].set_title("Word Count by Sentiment", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Sentiment", fontweight='bold')
axes[1, 1].set_ylabel("Word Count", fontweight='bold')

plt.tight_layout()
plt.savefig('08_tweet_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 08_tweet_length_analysis.png\n")


# ----------------------------
# 9️⃣ ENGAGEMENT METRICS ANALYSIS (NEW!)
# ----------------------------
print("=" * 60)
print("9️⃣ ENGAGEMENT METRICS ANALYSIS")
print("=" * 60)

# Calculate engagement metrics
df['total_engagement'] = df['likes'] + df['retweets'] + df['replies'] + df['quotes']
df['engagement_rate'] = df['total_engagement'] / (df['total_engagement'].max() + 1)  # Normalized

# Basic statistics
print("\n📊 Engagement Statistics:")
print(f"Average likes: {df['likes'].mean():.2f} (median: {df['likes'].median():.0f})")
print(f"Average retweets: {df['retweets'].mean():.2f} (median: {df['retweets'].median():.0f})")
print(f"Average replies: {df['replies'].mean():.2f} (median: {df['replies'].median():.0f})")
print(f"Average quotes: {df['quotes'].mean():.2f} (median: {df['quotes'].median():.0f})")
print(f"Average total engagement: {df['total_engagement'].mean():.2f} (median: {df['total_engagement'].median():.0f})")

# Identify zero-engagement tweets
zero_engagement = df[df['total_engagement'] == 0]
print(f"\n🔍 Zero Engagement Analysis:")
print(f"Tweets with zero engagement: {len(zero_engagement)} ({len(zero_engagement) / len(df) * 100:.1f}%)")
print(
    f"Tweets with any engagement: {len(df) - len(zero_engagement)} ({(len(df) - len(zero_engagement)) / len(df) * 100:.1f}%)")


# Define engagement tiers
def categorize_engagement(total_eng):
    if total_eng == 0:
        return 'No Engagement'
    elif total_eng <= 5:
        return 'Low (1-5)'
    elif total_eng <= 50:
        return 'Medium (6-50)'
    elif total_eng <= 500:
        return 'High (51-500)'
    else:
        return 'Viral (500+)'


df['engagement_tier'] = df['total_engagement'].apply(categorize_engagement)

engagement_dist = df['engagement_tier'].value_counts()
print(f"\n📊 Engagement Distribution:")
print(engagement_dist)

# High-engagement tweets (top 10%)
top_10_pct = df['total_engagement'].quantile(0.90)
high_engagement = df[df['total_engagement'] >= top_10_pct]
print(f"\n🔥 High Engagement Tweets (Top 10%):")
print(f"Threshold: {top_10_pct:.0f} total engagements")
print(f"Count: {len(high_engagement)} tweets")
print(f"Sentiment distribution in high-engagement tweets:")
print(high_engagement['sentiment'].value_counts())

# Create comprehensive engagement visualizations
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)

# 1. Engagement distribution (log scale for better visibility)
ax1 = fig.add_subplot(gs[0, 0])
engagement_metrics = ['likes', 'retweets', 'replies', 'quotes']
avg_values = [df[metric].mean() for metric in engagement_metrics]
median_values = [df[metric].median() for metric in engagement_metrics]

x = np.arange(len(engagement_metrics))
width = 0.35
bars1 = ax1.bar(x - width / 2, avg_values, width, label='Mean', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width / 2, median_values, width, label='Median', alpha=0.8, color='coral')

ax1.set_title("Average vs Median Engagement Metrics", fontsize=12, fontweight='bold')
ax1.set_ylabel("Count")
ax1.set_xticks(x)
ax1.set_xticklabels(engagement_metrics)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 2. Engagement tier distribution
ax2 = fig.add_subplot(gs[0, 1])
tier_order = ['No Engagement', 'Low (1-5)', 'Medium (6-50)', 'High (51-500)', 'Viral (500+)']
tier_data = engagement_dist.reindex(tier_order, fill_value=0)
colors_tier = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#9B59B6']
bars = ax2.barh(tier_order, tier_data.values, color=colors_tier, alpha=0.8)
ax2.set_title("Engagement Tier Distribution", fontsize=12, fontweight='bold')
ax2.set_xlabel("Number of Tweets")
for i, (bar, val) in enumerate(zip(bars, tier_data.values)):
    ax2.text(val + max(tier_data.values) * 0.01, i, f'{val} ({val / len(df) * 100:.1f}%)',
             va='center', fontweight='bold', fontsize=9)

# 3. Engagement tier pie chart
ax3 = fig.add_subplot(gs[0, 2])
ax3.pie(tier_data.values, labels=tier_order, autopct='%1.1f%%',
        colors=colors_tier, startangle=90)
ax3.set_title("Engagement Distribution (%)", fontsize=12, fontweight='bold')

# 4. Sentiment by engagement tier
ax4 = fig.add_subplot(gs[1, :])
engagement_sentiment = pd.crosstab(df['engagement_tier'], df['sentiment'], normalize='index') * 100
engagement_sentiment = engagement_sentiment.reindex(tier_order, fill_value=0)
engagement_sentiment = engagement_sentiment.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0)

engagement_sentiment.plot(kind='bar', stacked=True, ax=ax4,
                          color=get_sentiment_colors(['negative', 'neutral', 'positive']))
ax4.set_title("Sentiment Distribution by Engagement Tier", fontsize=14, fontweight='bold')
ax4.set_xlabel("Engagement Tier", fontweight='bold')
ax4.set_ylabel("Percentage (%)", fontweight='bold')
ax4.legend(title="Sentiment", loc='upper right')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Scatter: Total engagement vs sentiment (with jitter for visibility)
ax5 = fig.add_subplot(gs[2, 0])
for sentiment in ['positive', 'negative', 'neutral']:
    subset = df[df['sentiment'] == sentiment]
    # Add jitter to avoid overlapping points
    jitter_x = np.random.normal(0, 0.1, len(subset))
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    x_pos = [sentiment_map[sentiment]] * len(subset) + jitter_x
    ax5.scatter(x_pos, subset['total_engagement'],
                alpha=0.5, s=30, c=SENTIMENT_COLORS[sentiment], label=sentiment.capitalize())

ax5.set_yscale('log')
ax5.set_title("Engagement by Sentiment", fontsize=12, fontweight='bold')
ax5.set_ylabel("Total Engagement (log scale)", fontweight='bold')
ax5.set_xticks([-1, 0, 1])
ax5.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Box plot: Engagement metrics by sentiment
ax6 = fig.add_subplot(gs[2, 1])
sentiment_order_box = ['negative', 'neutral', 'positive']
# Use log scale for better visibility
df_log = df.copy()
df_log['log_engagement'] = np.log1p(df_log['total_engagement'])
sns.boxplot(data=df_log, x='sentiment', y='log_engagement',
            palette=get_sentiment_colors(sentiment_order_box),
            order=sentiment_order_box, ax=ax6)
ax6.set_title("Engagement Distribution by Sentiment", fontsize=12, fontweight='bold')
ax6.set_xlabel("Sentiment", fontweight='bold')
ax6.set_ylabel("Log(Total Engagement + 1)", fontweight='bold')

# 7. Top engaged tweets by sentiment
ax7 = fig.add_subplot(gs[2, 2])
top_by_sentiment = []
for sentiment in ['positive', 'negative', 'neutral']:
    top_tweet = df[df['sentiment'] == sentiment].nlargest(1, 'total_engagement')
    if len(top_tweet) > 0:
        top_by_sentiment.append({
            'sentiment': sentiment,
            'engagement': top_tweet['total_engagement'].values[0]
        })

if top_by_sentiment:
    sentiments = [item['sentiment'] for item in top_by_sentiment]
    engagements = [item['engagement'] for item in top_by_sentiment]
    colors_bar = [SENTIMENT_COLORS[s] for s in sentiments]
    ax7.bar(sentiments, engagements, color=colors_bar, alpha=0.8)
    ax7.set_title("Highest Engagement Tweet by Sentiment", fontsize=12, fontweight='bold')
    ax7.set_ylabel("Total Engagement", fontweight='bold')
    ax7.set_xlabel("Sentiment", fontweight='bold')
    for i, v in enumerate(engagements):
        ax7.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

# 8. Engagement over time (daily average)
ax8 = fig.add_subplot(gs[3, :])
daily_engagement = df.groupby('date_only').agg({
    'likes': 'mean',
    'retweets': 'mean',
    'total_engagement': 'mean'
}).reset_index()

ax8.plot(daily_engagement['date_only'], daily_engagement['likes'],
         marker='o', label='Avg Likes', linewidth=2, color='#E74C3C')
ax8.plot(daily_engagement['date_only'], daily_engagement['retweets'],
         marker='s', label='Avg Retweets', linewidth=2, color='#3498DB')
ax8.plot(daily_engagement['date_only'], daily_engagement['total_engagement'],
         marker='^', label='Avg Total Engagement', linewidth=2.5, color='#2ECC71')

for event_date in KEY_EVENTS.keys():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_engagement['date_only'].values:
        ax8.axvline(x=event_dt, color='red', linestyle='--', linewidth=2, alpha=0.6)

ax8.set_title("Daily Average Engagement Trends", fontsize=14, fontweight='bold')
ax8.set_xlabel("Date", fontweight='bold')
ax8.set_ylabel("Average Engagement", fontweight='bold')
ax8.legend(loc='best')
ax8.grid(True, alpha=0.3)

plt.suptitle("ENGAGEMENT METRICS ANALYSIS", fontsize=16, fontweight='bold', y=0.995)
plt.savefig('09a_engagement_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✅ Saved: 09a_engagement_analysis.png")

# Print detailed insights
print("\n" + "=" * 60)
print("🔍 ENGAGEMENT INSIGHTS")
print("=" * 60)

print("\n1️⃣ Most Engaged Tweets:")
top_engaged = df.nlargest(10, 'total_engagement')[
    ['text', 'sentiment', 'likes', 'retweets', 'replies', 'total_engagement']]
for idx, (i, row) in enumerate(top_engaged.iterrows(), 1):
    print(f"\n{idx}. [{row['sentiment'].upper()}] Total: {row['total_engagement']:.0f}")
    print(f"   Likes: {row['likes']:.0f} | RT: {row['retweets']:.0f} | Replies: {row['replies']:.0f}")
    print(f"   Text: {row['text'][:100]}...")

print("\n2️⃣ Sentiment Analysis of High-Engagement Tweets:")
for tier in ['Viral (500+)', 'High (51-500)', 'Medium (6-50)']:
    if tier in df['engagement_tier'].values:
        tier_df = df[df['engagement_tier'] == tier]
        print(f"\n{tier}:")
        print(f"   Total tweets: {len(tier_df)}")
        print(f"   Sentiment: {tier_df['sentiment'].value_counts().to_dict()}")
        print(f"   Avg engagement: {tier_df['total_engagement'].mean():.1f}")

print("\n3️⃣ Zero-Engagement Analysis:")
if len(zero_engagement) > 0:
    print(f"Zero-engagement tweets by sentiment:")
    print(zero_engagement['sentiment'].value_counts())
    print(f"\nZero-engagement tweets by topic:")
    print(zero_engagement['primary_topic'].value_counts().head(5))

# Correlation analysis
print("\n4️⃣ Engagement Correlations:")
print("\nCorrelation between engagement metrics:")
engagement_corr = df[['likes', 'retweets', 'replies', 'quotes']].corr()
print(engagement_corr)

print("\n5️⃣ Engagement by Topic:")
topic_engagement = df.groupby('primary_topic').agg({
    'total_engagement': ['mean', 'median', 'max'],
    'likes': 'mean',
    'retweets': 'mean'
}).round(2)
topic_engagement = topic_engagement.sort_values(('total_engagement', 'mean'), ascending=False)
print(topic_engagement.head(10))

# Statistical test: Sentiment vs Engagement
print("\n6️⃣ Statistical Analysis:")
print("Average total engagement by sentiment:")
for sentiment in ['positive', 'negative', 'neutral']:
    avg_eng = df[df['sentiment'] == sentiment]['total_engagement'].mean()
    med_eng = df[df['sentiment'] == sentiment]['total_engagement'].median()
    print(f"   {sentiment.capitalize()}: Mean={avg_eng:.2f}, Median={med_eng:.0f}")

# Engagement rate analysis (for tweets with any engagement)
engaged_tweets = df[df['total_engagement'] > 0]
print(f"\n7️⃣ Engaged Tweets Analysis ({len(engaged_tweets)} tweets):")
print(f"Average engagement: {engaged_tweets['total_engagement'].mean():.2f}")
print(f"Median engagement: {engaged_tweets['total_engagement'].median():.0f}")
print(f"Most common sentiment: {engaged_tweets['sentiment'].mode()[0]}")

print("\n✅ Engagement analysis complete!\n")

# Analyze topics around specific events
event_topics = {}

for event_name, event_info in event_windows.items():
    event_date = datetime.strptime(event_info['date'], '%Y-%m-%d').date()
    start_date = event_date - timedelta(days=event_info['before'])
    end_date = event_date + timedelta(days=event_info['after'])

    mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
    event_df = df[mask]

    if len(event_df) == 0:
        continue

    topic_dist = event_df['primary_topic'].value_counts()
    event_topics[event_name] = topic_dist

    print(f"\n📌 {event_name}")
    print(f"   Top topics during event window:")
    print(topic_dist.head(5))

# Visualize event-specific topics
if len(event_topics) > 0:
    fig, axes = plt.subplots(1, len(event_topics), figsize=(8 * len(event_topics), 6))
    if len(event_topics) == 1:
        axes = [axes]

    for idx, (event_name, topic_dist) in enumerate(event_topics.items()):
        top_topics = topic_dist.head(8)
        axes[idx].barh(top_topics.index, top_topics.values, color='teal', alpha=0.8)
        axes[idx].set_title(f"Top Topics: {event_name}", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel("Number of Tweets", fontweight='bold')
        axes[idx].set_ylabel("Topic", fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')

        for i, v in enumerate(top_topics.values):
            axes[idx].text(v + 0.5, i, str(v), va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('09_event_specific_topics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ Saved: 09_event_specific_topics.png\n")
else:
    print("\n⚠️ No event data available for visualization\n")

# ----------------------------
# 9️⃣ EVENT-SPECIFIC TOPIC ANALYSIS
# ----------------------------
print("=" * 60)
print("9️⃣ EVENT-SPECIFIC TOPIC ANALYSIS")
print("=" * 60)

# Analyze topics around specific events
event_topics = {}

for event_name, event_info in event_windows.items():
    event_date = datetime.strptime(event_info['date'], '%Y-%m-%d').date()
    start_date = event_date
    end_date = event_date

    mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
    event_df = df[mask]

    if len(event_df) == 0:
        continue

    topic_dist = event_df['primary_topic'].value_counts()
    event_topics[event_name] = topic_dist

    print(f"\n📌 {event_name}")
    print(f"   Top topics during event window:")
    print(topic_dist.head(5))

# Visualize event-specific topics
fig, axes = plt.subplots(1, len(event_topics), figsize=(8 * len(event_topics), 6))
if len(event_topics) == 1:
    axes = [axes]

for idx, (event_name, topic_dist) in enumerate(event_topics.items()):
    top_topics = topic_dist.head(8)
    axes[idx].barh(top_topics.index, top_topics.values, color='teal', alpha=0.8)
    axes[idx].set_title(f"Top Topics: {event_name}", fontsize=14, fontweight='bold')
    axes[idx].set_xlabel("Number of Tweets", fontweight='bold')
    axes[idx].set_ylabel("Topic", fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(top_topics.values):
        axes[idx].text(v + 0.5, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('09_event_specific_topics.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✅ Saved: 09_event_specific_topics.png\n")


# ----------------------------
# 🔟 COMPREHENSIVE SUMMARY REPORT
# ----------------------------
print("=" * 60)
print("🔟 COMPREHENSIVE SUMMARY REPORT")
print("=" * 60)

summary_report = {
    'Dataset Overview': {
        'Total Tweets': len(df),
        'Date Range': f"{df['date'].min().date()} to {df['date'].max().date()}",
        'Days Covered': len(df['date_only'].unique())
    },
    'Sentiment Distribution': {
        'Positive': f"{len(df[df['sentiment'] == 'positive'])} ({len(df[df['sentiment'] == 'positive']) / len(df) * 100:.1f}%)",
        'Negative': f"{len(df[df['sentiment'] == 'negative'])} ({len(df[df['sentiment'] == 'negative']) / len(df) * 100:.1f}%)",
        'Neutral': f"{len(df[df['sentiment'] == 'neutral'])} ({len(df[df['sentiment'] == 'neutral']) / len(df) * 100:.1f}%)"
    },
    'Top Topics': dict(topic_counts.head(5)),
    'Tweet Statistics': {
        'Avg Character Count': f"{df['char_count'].mean():.1f}",
        'Avg Word Count': f"{df['word_count'].mean():.1f}",
        'Peak Day': f"{daily_total.idxmax()} ({daily_total.max()} tweets)"
    }
}

print("\n📊 SUMMARY STATISTICS:")
for section, data in summary_report.items():
    print(f"\n{section}:")
    for key, value in data.items():
        print(f"  {key}: {value}")

# Save summary to CSV
summary_df = pd.DataFrame({
    'Metric': ['Total Tweets', 'Positive %', 'Negative %', 'Neutral %',
               'Avg Characters', 'Avg Words', 'Most Common Topic', 'Days Covered'],
    'Value': [
        len(df),
        f"{len(df[df['sentiment'] == 'positive']) / len(df) * 100:.1f}%",
        f"{len(df[df['sentiment'] == 'negative']) / len(df) * 100:.1f}%",
        f"{len(df[df['sentiment'] == 'neutral']) / len(df) * 100:.1f}%",
        f"{df['char_count'].mean():.1f}",
        f"{df['word_count'].mean():.1f}",
        topic_counts.index[0],
        len(df['date_only'].unique())
    ]
})

summary_df.to_csv('10_summary_statistics.csv', index=False)
print("\n✅ Saved: 10_summary_statistics.csv")

# Create summary visualization dashboard
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Sentiment pie
ax1 = fig.add_subplot(gs[0, 0])
pie_data = [sentiment_counts.get(s, 0) for s in sentiment_order]
ax1.pie(pie_data, labels=[s.capitalize() for s in sentiment_order],
        autopct='%1.1f%%', colors=pie_colors, startangle=140)
ax1.set_title("Overall Sentiment", fontsize=12, fontweight='bold')

# 2. Daily trends mini
ax2 = fig.add_subplot(gs[0, 1:])
for i, sentiment in enumerate(['negative', 'neutral', 'positive']):
    ax2.plot(daily_sentiment.index, daily_sentiment[sentiment],
             label=sentiment.capitalize(), color=line_colors[i], linewidth=2)
for event_date in KEY_EVENTS.keys():
    event_dt = datetime.strptime(event_date, '%Y-%m-%d').date()
    if event_dt in daily_sentiment.index:
        ax2.axvline(x=event_dt, color='red', linestyle='--', alpha=0.5)
ax2.set_title("Sentiment Timeline", fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Top topics
ax3 = fig.add_subplot(gs[1, 0])
top_5 = topic_counts.head(5)
ax3.barh(top_5.index, top_5.values, color='steelblue', alpha=0.8)
ax3.set_title("Top 5 Topics", fontsize=12, fontweight='bold')
ax3.set_xlabel("Tweets")

# 4. Topic-sentiment heatmap
ax4 = fig.add_subplot(gs[1, 1:])
top_topics_for_heatmap = topic_counts.head(7).index
heatmap_data = topic_sentiment_counts.loc[top_topics_for_heatmap]
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
ax4.set_title("Topic-Sentiment Heatmap", fontsize=12, fontweight='bold')

# 5. Length distribution
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(df['word_count'], bins=30, color='coral', alpha=0.7, edgecolor='black')
ax5.axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2)
ax5.set_title("Tweet Length Distribution", fontsize=12, fontweight='bold')
ax5.set_xlabel("Words")
ax5.set_ylabel("Frequency")

# 6. Day of week
ax6 = fig.add_subplot(gs[2, 1])
ax6.bar(dow_counts.index, dow_counts.values, color='teal', alpha=0.8)
ax6.set_title("Tweets by Day of Week", fontsize=12, fontweight='bold')
ax6.tick_params(axis='x', rotation=45, labelsize=8)

# 7. Key metrics text
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
metrics_text = f"""
KEY METRICS

Total Tweets: {len(df):,}

Sentiment Breakdown:
  Positive: {len(df[df['sentiment'] == 'positive']):,}
  Negative: {len(df[df['sentiment'] == 'negative']):,}
  Neutral: {len(df[df['sentiment'] == 'neutral']):,}

Avg Length: {df['word_count'].mean():.1f} words

Peak Day: {daily_total.idxmax()}
Peak Volume: {daily_total.max()} tweets

Most Discussed: {topic_counts.index[0].replace('_', ' ').title()}
"""
ax7.text(0.1, 0.9, metrics_text, fontsize=10, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("DUOLINGO SENTIMENT ANALYSIS - EXECUTIVE DASHBOARD",
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('10_executive_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: 10_executive_dashboard.png")

# ----------------------------
# FINAL SUMMARY
# ----------------------------
print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print("\n📁 Generated Visualizations:")
viz_files = [
    "01_sentiment_distribution.png",
    "02_event_timeline_analysis.png",
    "03_event_impact_analysis.png",
    "04_topic_analysis.png",
    "05_wordclouds_by_sentiment.png",
    "06_ngram_analysis.png",
    "07_temporal_patterns.png",
    "08_tweet_length_analysis.png",
    "09_event_specific_topics.png",
    "10_executive_dashboard.png",
    "10_summary_statistics.csv"
]

for i, file in enumerate(viz_files, 1):
    print(f"  {i:2}. {file}")

print("\n🎯 Key Findings:")
print(f"  • Dataset covers {len(df['date_only'].unique())} days")
print(f"  • Most common sentiment: {sentiment_counts.idxmax().upper()}")
print(f"  • Most discussed topic: {topic_counts.index[0].replace('_', ' ').title()}")
print(f"  • Peak activity: {daily_total.idxmax()} with {daily_total.max()} tweets")
print(f"  • Average tweet length: {df['word_count'].mean():.1f} words")

# Calculate sentiment change around events
if len(event_windows) > 0:
    print("\n📊 Event Impact Summary:")
    for event_name, event_info in event_windows.items():
        event_date = datetime.strptime(event_info['date'], '%Y-%m-%d').date()

        before_mask = df['date_only'] < event_date
        after_mask = df['date_only'] >= event_date

        if before_mask.sum() > 0 and after_mask.sum() > 0:
            before_neg = (df[before_mask]['sentiment'] == 'negative').sum() / before_mask.sum() * 100
            after_neg = (df[after_mask]['sentiment'] == 'negative').sum() / after_mask.sum() * 100
            change = after_neg - before_neg

            print(f"  • {event_name}:")
            print(f"    Negative sentiment changed by {change:+.1f}% after event")

print("\n" + "=" * 60)
print("Thank you for using Duolingo Sentiment Analysis!")
print("=" * 60)