# Duolingo Sentiment Analysis

**Duolingo Sentiment Analysis** is an exploratory NLP and data analytics project that examines public sentiment toward Duolingo using Twitter data. The project combines multiple sentiment classification approaches, text preprocessing pipelines, and exploratory analysis to uncover **user pain points, product risks, and the impact of external events** on brand perception.

> Note: This repository reflects an **exploratory, research-oriented workflow** rather than a production-ready pipeline. The focus is on insight discovery and comparative analysis of NLP techniques.

---

## Project Overview

This project analyzes **5,229 tweets mentioning Duolingo**, collected between **October 18–23, 2025**, to answer the following questions:

- How do users emotionally perceive Duolingo overall?
- Which product features generate the strongest negative or positive reactions?
- How do external incidents (e.g., infrastructure outages) affect sentiment dynamics?
- Which sentiment analysis techniques yield consistent signals across noisy social media data?

The analysis classifies tweets into **positive, neutral, and negative sentiment** categories and investigates temporal, topical, and engagement-based patterns.

---

## Key Results & Insights

### Product Risk Identification
- **Duolingo Energy system** emerged as a major pain point, with **~62% negative sentiment**, signaling strong user dissatisfaction related to monetization and usage limits.

### Crisis Impact Analysis
- A major **AWS outage** during the data collection period accounted for **~47% of total tweet volume**.
- Sentiment sharply shifted toward negative during the outage window, followed by partial recovery, clearly visible in temporal sentiment plots.

### Sentiment Distribution
- Negative sentiment dominated discussions during high-engagement events.
- Neutral sentiment prevailed outside crisis periods, while positive sentiment clustered around learning achievements and app usability.

### Analytical Value
- Demonstrated how **event-driven spikes** can distort overall sentiment metrics if not analyzed separately.
- Highlighted the importance of contextual segmentation in brand sentiment analysis.

---

## Methodology

### 1. Data Collection
- Tweets containing Duolingo-related keywords and hashtags.
- Dataset includes tweet text, timestamps, and engagement metadata.

### 2. Text Preprocessing
Multiple preprocessing pipelines were implemented to compare outcomes:
- Tokenization, normalization, punctuation and URL removal
- NLTK-based preprocessing
- spaCy-based preprocessing

### 3. Sentiment Classification
Several sentiment analysis approaches were tested:
- **Naive Bayes classifier**
- **Maximum Entropy (MaxEnt) classifier**
- **TextBlob sentiment**
- **VADER sentiment analysis**

This enabled comparison of model behavior across different linguistic assumptions.

### 4. Exploratory Data Analysis (EDA)
- Temporal sentiment trends
- Topic and n-gram frequency analysis
- Word clouds by sentiment category
- Engagement metrics vs. sentiment polarity

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Kateroook/duolingo-sentiment-analysis.git
cd duolingo-sentiment-analysis
```
2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install pandas numpy nltk spacy matplotlib seaborn
```
4. Run preprocessing and analysis scripts:
```bash
python preprocessing_nltk.py
python eda_duolingo.py
```
>Some results and visualizations are already generated and included in the repository ^^

--- 
## Limitations
- The project is exploratory and not organized as a production ML pipeline.
- Dataset size and time window are limited to a short observation period.
- Twitter data may contain noise, sarcasm, and event-driven bias.

---
