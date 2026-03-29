import pandas as pd
import datetime
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from transformers import pipeline
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n********************* 📅 Starting sentiment analysis - {now} **********************\n")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("articles.csv", encoding='utf-8-sig')
print(f"✅ Loaded {len(df)} articles\n")

# ── 1. LANGUAGE DETECTION AND TRANSLATION TO ENGLISH ──────────────────────────────────────────────────
print("🌍 Detecting languages...")

def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"

def translate_to_english(text, lang):
    """Translates Spanish, Portuguese, and German titles to English for NRC scoring."""
    if lang in ['es', 'pt', 'de', 'fr']:  # Spanish, Portuguese, German, French
        try:
            return GoogleTranslator(source=lang, target='en').translate(str(text)[:500])
        except Exception:
            return text  # fallback to original if translation fails
    return text

df['language'] = df['title'].apply(detect_language)
print(df.groupby(['region', 'language']).size().unstack(fill_value=0))
print("\n🔄 Translating Spanish/Portuguese/German/French titles to English for NRC analysis...")
df['title_for_nrc'] = df.apply(
    lambda row: translate_to_english(row['title'], row['language']), axis=1
)
print("✅ Translation complete!")

# # ── 2.1 HUGGINGFACE MULTILINGUAL SENTIMENT XLM-ROBERTA ──────────────────────────────────
# print("\n🤗 Running multilingual sentiment analysis Cardiff NLP...")
# sentiment_model = pipeline(
#     "sentiment-analysis",
#     model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
#     truncation=True,
#     max_length=128
# )

# def get_sentiment(text):
#     try:
#         result = sentiment_model(str(text)[:512])[0]
#         return result['label'].lower(), round(result['score'], 4)
#     except Exception:
#         return "unknown", None
# df['sentiment'], df['sentiment_score'] = zip(*df['title'].apply(get_sentiment))
# print("\n📊 XLM-Roberta Sentiment breakdown by region:")
# print(df.groupby(['region', 'sentiment']).size().unstack(fill_value=0))

# ── 2.2 HUGGINGFACE MULTILINGUAL SENTIMENT BERT MULTILINGUAL ──────────────────────────────────
print("\n🤗 Running multilingual sentiment analysis NLP Town...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=128
)

def get_sentiment(text):
    try:
        result = sentiment_model(str(text)[:512])[0]
        stars = int(result['label'][0])  # extracts the number from "1 star", "3 stars" etc.
        # Convert stars to sentiment label
        if stars <= 2:
            label = "negative"
        elif stars == 3:
            label = "neutral"
        else:
            label = "positive"
        return label, round(result['score'], 4)
    except Exception:
        return "unknown", None


df['sentiment'], df['sentiment_score'] = zip(*df['title'].apply(get_sentiment))
print("\n📊 BERT Sentiment breakdown by region:")
print(df.groupby(['region', 'sentiment']).size().unstack(fill_value=0))

# ── 3. NRC LEXICON (EMOTION ANALYSIS) ─────────────────────────────────────
print("\n😡 Running NRC emotion analysis...")

# Download NRC lexicon
import urllib.request, os
NRC_URL = "https://raw.githubusercontent.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
NRC_FILE = "nrc_lexicon.txt"

if not os.path.exists(NRC_FILE):
    print("  Downloading NRC lexicon...")
    urllib.request.urlretrieve(NRC_URL, NRC_FILE)

# Parse NRC lexicon
nrc = {}
with open(NRC_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word, emotion, present = parts
            if int(present) == 1:
                nrc.setdefault(word, []).append(emotion)

EMOTIONS = ['anger', 'fear', 'joy', 'trust', 'sadness',
            'surprise', 'disgust', 'anticipation']

def get_emotions(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    counts = {e: 0 for e in EMOTIONS}
    for word in words:
        for emotion in nrc.get(word, []):
            if emotion in counts:
                counts[emotion] += 1
    return counts

emotion_results = df['title_for_nrc'].apply(get_emotions)
emotion_df = pd.DataFrame(emotion_results.tolist())
df = pd.concat([df, emotion_df], axis=1)

print("\n😤 Average emotion scores by region:")
print(df.groupby('region')[EMOTIONS].mean().round(3))

# ── 4. BIGRAMS (FRAMING ANALYSIS) ─────────────────────────────────────────
print("\n🔤 Running bigram framing analysis...")

# Stopwords for English, Spanish, German, French
stop_en = set(stopwords.words('english'))
stop_es = set(stopwords.words('spanish'))
stop_de = set(stopwords.words('german'))
stop_fr = set(stopwords.words('french'))
all_stopwords = stop_en | stop_es | stop_de | stop_fr

def get_bigrams(texts, top_n=10):
    words_list = []
    for text in texts:
        words = re.findall(r'\b\w+\b', str(text).lower())
        words = [w for w in words if w not in all_stopwords and len(w) > 2]
        words_list.extend(list(ngrams(words, 2)))
    return Counter(words_list).most_common(top_n)

print("\n🔤 Top bigrams by region:")
for region in df['region'].unique():
    region_titles = df[df['region'] == region]['title_for_nrc']
    print(f"\n  {region}:")
    for bigram, count in get_bigrams(region_titles):
        print(f"    {' '.join(bigram)}: {count}")

# ── 5. LDA TOPIC MODELING ──────────────────────────────────────────────────
print("\n📑 Running LDA topic modeling...")

def clean_text(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    return ' '.join([w for w in words if w not in all_stopwords and len(w) > 2])

print("\n📑 Top topics per region (LDA):")
for region in df['region'].unique():
    region_titles = df[df['region'] == region]['title_for_nrc'].apply(clean_text)

    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(region_titles)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)

    print(f"\n  {region}:")
    for i, topic in enumerate(lda.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-6:-1]]
        print(f"    Topic {i+1}: {', '.join(top_words)}")

# ── Save results ───────────────────────────────────────────────────────────
df.to_csv("articles_with_sentiment.csv", index=False, encoding='utf-8-sig')
print("\n✅ All analysis complete! Saved to articles_with_sentiment.csv")