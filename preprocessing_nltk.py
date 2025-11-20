import pandas as pd
import re
import nltk
import html

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.util import mark_negation
#
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# ----------------------------
# Load the dataset
# ----------------------------
df = pd.read_csv("tweets_to_train.csv")

SHORTFORM_MAP = {
    # Common spoken contractions
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "ain't": "is not",
    "imma": "i am going to",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
    "outta": "out of",
    "lotta": "lot of",
    "tryna": "trying to",
    "hafta": "have to",
    "needa": "need to",
    "dunno": "do not know",
    "cuz": "because",
    "cause": "because",
    "thru": "through",
    "ya": "you",
    "yall": "you all",
    "ya'll": "you all",
    "y'all": "you all",
    "im": "i am",
    "id": "i would",
    "ill": "i will",
    "ive": "i have",
    "cant": "cannot",
    "wont": "will not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "havent":"have not",
    "hasnt":"has not",
    "shoulda": "should have",
    "woulda": "would have",
    "coulda": "could have",
    "mighta": "might have",
    "aint": "is not",


    # Internet/text slang
    "ur": "your",
    "r": "are",
    "re": "are",
    "s": "is",
    "m": "am",
    "u": "you",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "bc": "because",
    "bcz": "because",
    "bcos": "because",
    "jk": "just kidding",
    "gg": "good game",
    "af": "as fuck",
    "tf": "the fuck",
    "tysm": "thank you so much",
    "yw": "you are welcome",
    "tho": "though",
    "bcuz": "because"
}

CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

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

    # Remove contractions
    for c, full in CONTRACTIONS.items():
        text = re.sub(c, full, text)

    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
df['text_cleaned'] = df['text_cleaned'].apply(remove_noise)

# ----------------------------
# Normalization (Lower Case + Stopword Removal + Lemmatization)
# ----------------------------

stop_words = set(stopwords.words('english'))
# Remove negations and important modifiers from stopwords
# negation_words = {'no', 'not', 'nor', 'never', 'none', 'nothing', 'nowhere', 'neither', 'hardly', 'barely', 'scarcely'}
stop_words = stop_words - {'down','no', 'not', 'never'}

lemmatizer = WordNetLemmatizer()

# convert NLTK POS to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag == "PRP" or tag == "PRP$":
        return None  # skip pronouns
    else:
        return None


def expand_shortforms(text):
    # Sort keys by length (descending) to prevent partial matches (e.g. "u" in "ur")
    sorted_shortforms = sorted(SHORTFORM_MAP.items(), key=lambda x: len(x[0]), reverse=True)

    for short, full in sorted_shortforms:
        # Use regex with word boundaries, case-insensitive
        pattern = re.compile(rf"\b{re.escape(short)}\b", flags=re.IGNORECASE)
        text = pattern.sub(full, text)
    return text


def normalize_tokenize(text):
    # Lowercase
    text = text.lower()

    # Expand short forms
    text = expand_shortforms(text)
    # Tokenize
    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # POS tagging
    pos_tags = pos_tag(tokens)

    lemmas = []
    for w, p in pos_tags:
        wn_pos = get_wordnet_pos(p)
        if wn_pos:
            lemmas.append(lemmatizer.lemmatize(w, wn_pos))
        else:
            lemmas.append(w.lower())
    # lemmas = []
    # for i, (word, tag) in enumerate(pos_tags):
    #     wn_pos = get_wordnet_pos(tag)
    #     lemma = lemmatizer.lemmatize(word, wn_pos)
    #
    #     # --- improved 'ing' fallback for mis-tagged verbs ---
    #     if word.endswith("ing"):
    #         prev_tags = [t for _, t in pos_tags[max(0, i-2):i]]
    #         # if previous token is a pronoun/adverb/auxiliary, likely a verb
    #         if any(pt.startswith(("V", "RB", "PRP")) for pt in prev_tags):
    #             verb_lemma = lemmatizer.lemmatize(word, wordnet.VERB)
    #             if verb_lemma != word:
    #                 lemma = verb_lemma
    #
    #     lemmas.append(lemma)
    #
    # # now remove stopwords and short tokens from the lemmas
    # lemmas = [w for w in lemmas if w not in stop_words]

    return lemmas

df['tokens'] = df['text_cleaned'].apply(normalize_tokenize)
df['clean_text_joined'] = df['tokens'].apply(lambda x: ' '.join(x))

# ----------------------------
# Save preprocessed data
# ----------------------------
df.to_csv("tweets_to_train.csv", index=False, encoding='utf-8-sig')

print("✅ Data preprocessing complete.")
# print(df[['text', 'text_cleaned', 'tokens']].head(5))