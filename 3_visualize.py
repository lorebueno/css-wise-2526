from tracemalloc import stop

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("articles_with_sentiment.csv", encoding='utf-8-sig')
df['region'] = df['region'].str.replace('_', ' ')

REGIONS = ['Europe', 'Latin America', 'US', 'Venezuela']
EMOTIONS = ['anger', 'fear', 'joy', 'trust', 'sadness', 'surprise', 'disgust', 'anticipation']
COLORS = {
    'Europe':        '#4C72B0',
    'Latin America': '#DD8452',
    'US':            '#55A868',
    'Venezuela':     '#C44E52'
}

print("📊 Building visualizations...")

# ── 1. SENTIMENT BREAKDOWN BAR CHART ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

sentiment_data = df.groupby(['region', 'sentiment']).size().unstack(fill_value=0)
sentiment_data = sentiment_data[['negative', 'neutral', 'positive']]

x = np.arange(len(REGIONS))
width = 0.25
sentiment_colors = ['#C44E52', '#8C8C8C', '#55A868']

for i, (sentiment, color) in enumerate(zip(['negative', 'neutral', 'positive'], sentiment_colors)):
    values = [sentiment_data.loc[r, sentiment] if r in sentiment_data.index else 0 for r in REGIONS]
    bars = ax.bar(x + i * width, values, width, label=sentiment.capitalize(), color=color, alpha=0.85)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(val), ha='center', va='bottom', fontsize=9)

ax.set_xticks(x + width)
ax.set_xticklabels(REGIONS, fontsize=11)
ax.set_ylabel('Number of Articles', fontsize=11)
ax.set_title('Sentiment Breakdown by Region\nTrump & Venezuela Coverage (02-10 Jan 2026)', fontsize=13, fontweight='bold')
ax.legend(title='Sentiment', fontsize=10)
ax.set_ylim(0, max(sentiment_data.max()) + 8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('chart_sentiment.png', dpi=150)
plt.close()
print("  ✅ chart_sentiment.png")

# ── 2. EMOTION RADAR CHART ─────────────────────────────────────────────────
emotion_means = df.groupby('region')[EMOTIONS].mean()

angles = np.linspace(0, 2 * np.pi, len(EMOTIONS), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for region in REGIONS:
    values = emotion_means.loc[region].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=3, label=region, color=COLORS[region])
    ax.fill(angles, values, alpha=0.12, color=COLORS[region])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([e.capitalize() for e in EMOTIONS], fontsize=11, fontweight='bold')
ax.tick_params(pad=12)  # adds spacing between labels and chart edge
ax.set_title('Emotional Tone by Region\nNRC Lexicon Analysis', fontsize=13,
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
plt.savefig('chart_emotions_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ chart_emotions_radar.png")

# ── 3. EMOTION HEATMAP ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

emotion_means_plot = emotion_means.loc[REGIONS]
im = ax.imshow(emotion_means_plot.values, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(EMOTIONS)))
ax.set_xticklabels([e.capitalize() for e in EMOTIONS], fontsize=11)
ax.set_yticks(range(len(REGIONS)))
ax.set_yticklabels(REGIONS, fontsize=11)

for i in range(len(REGIONS)):
    for j in range(len(EMOTIONS)):
        val = emotion_means_plot.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
               fontsize=10, color='black' if val < 0.5 else 'white')

plt.colorbar(im, ax=ax, label='Average Score')
ax.set_title('Emotion Intensity Heatmap by Region\nNRC Lexicon (translated to English)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_emotions_heatmap.png', dpi=150)
plt.close()
print("  ✅ chart_emotions_heatmap.png")

# ── 4. TOP BIGRAMS PER REGION ──────────────────────────────────────────────
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
import re

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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, region in enumerate(REGIONS):
    region_titles = df[df['region'] == region]['title_for_nrc']
    bigrams = get_bigrams(region_titles)
    labels = [' '.join(b[0]) for b in bigrams][::-1]
    values = [b[1] for b in bigrams][::-1]

    axes[idx].barh(labels, values, color=COLORS[region], alpha=0.85)
    axes[idx].set_title(f'{region}', fontsize=12, fontweight='bold', color=COLORS[region])
    axes[idx].set_xlabel('Frequency', fontsize=10)
    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    for i, v in enumerate(values):
        axes[idx].text(v + 0.05, i, str(v), va='center', fontsize=9)

fig.suptitle('Top 10 Bigrams by Region - Framing Analysis \n (translated to English)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_bigrams.png', dpi=150)
plt.close()
print("  ✅ chart_bigrams.png")

# ── 5. LDA TOPICS TABLE ────────────────────────────────────────────────────
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    return ' '.join([w for w in words if w not in all_stopwords and len(w) > 2])

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for idx, region in enumerate(REGIONS):
    region_titles = df[df['region'] == region]['title_for_nrc'].apply(clean_text)
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(region_titles)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    topic_data = []
    for i, topic in enumerate(lda.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-6:-1]]
        topic_data.append([f'Topic {i+1}', ', '.join(top_words)])

    axes[idx].axis('off')
    table = axes[idx].table(
        cellText=topic_data,
        colLabels=['Topic', 'Top Words'],
        cellLoc='left',
        loc='center',
        colWidths=[0.2, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor(COLORS[region])
        table[0, j].set_text_props(color='white', fontweight='bold')

    axes[idx].set_title(f'{region}', fontsize=12,
                        fontweight='bold', color=COLORS[region])

fig.suptitle('LDA Topic Modeling by Region\n(translated to English)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_lda_topics.png', dpi=150)
plt.close()
print("  ✅ chart_lda_topics.png")

print("\n🎉 All charts saved!")
print("  → chart_sentiment.png")
print("  → chart_emotions_radar.png")
print("  → chart_emotions_heatmap.png")
print("  → chart_bigrams.png")
print("  → chart_lda_topics.png")

# ── 6. WORD CLOUDS PER REGION ──────────────────────────────────────────────
from wordcloud import WordCloud

print("\n☁️  Generating word clouds...")

# Install wordcloud if needed: pip install wordcloud
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, region in enumerate(REGIONS):
    region_titles = df[df['region'] == region]['title_for_nrc']
    
    # Combine all titles into one text blob
    text = ' '.join(region_titles.dropna().astype(str))
    
    # Clean stopwords
    clean_words = re.findall(r'\b\w+\b', text.lower())
    clean_text = ' '.join([w for w in clean_words 
                           if w not in all_stopwords and len(w) > 2])
    
    # Generate word cloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=lambda *args, **kwargs: COLORS[region],
        max_words=80,
        collocations=False,
        prefer_horizontal=0.8
    ).generate(clean_text)
    
    axes[idx].imshow(wc, interpolation='bilinear')
    axes[idx].axis('off')
    axes[idx].set_title(region, fontsize=14, fontweight='bold',
                        color=COLORS[region], pad=10)

fig.suptitle('Word Clouds by Region\nTrump & Venezuela Coverage (Jan 2–10, 2026)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_wordclouds.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ chart_wordclouds.png")