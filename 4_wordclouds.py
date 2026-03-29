from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
import re

stop_en = set(stopwords.words('english'))
stop_es = set(stopwords.words('spanish'))
stop_de = set(stopwords.words('german'))
all_stopwords = stop_en | stop_es | stop_de

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

print("\n☁️  Generating word clouds...")

for region in REGIONS:
    region_titles = df[df['region'] == region]['title_for_nrc']
    
    text = ' '.join(region_titles.dropna().astype(str))
    clean_words = re.findall(r'\b\w+\b', text.lower())
    clean_text = ' '.join([w for w in clean_words 
                           if w not in all_stopwords and len(w) > 2])
    
    wc = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        color_func=lambda *args, **kwargs: COLORS[region],
        max_words=80,
        collocations=False,
        prefer_horizontal=0.8
    ).generate(clean_text)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{region}\nWord Cloud — Trump & Venezuela Coverage (Jan 2–10, 2026)',
                 fontsize=16, fontweight='bold', color=COLORS[region], pad=15)
    
    # Save with region name, replacing space with underscore for filename
    filename = f"wordcloud_{region.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {filename}")