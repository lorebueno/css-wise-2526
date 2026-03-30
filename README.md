# Framing and Sentiment Analysis of Headlines on US Intervention in Venezuela

A computational text analysis project examining how news headlines about US intervention on Venezuela are framed across different regional media outlets, using sentiment analysis and topic modeling.

## Project Description

This project collects and analyzes news headlines related to the U.S.–Venezuela intervention occurred on Jan/2026 from four regional media contexts: the United States, Venezuela, Latin America, and Europe. The goal is to identify differences in emotional tone, sentiment polarity, and recurring themes across these regions using a combination of lexicon-based and transformer-based NLP methods.

## Research Goals

- Compare sentiment polarity in news coverage across four regional outlets
- Sentiment analysis: model-based sentiment scoring
- Computational goals: comparative of results for two NLP models trained for multilingual purposes
- Emotional analysis: identify dominant emotions using the NRC Emotion Lexicon tone classification.
- Extract recurring topics through bigram analysis and LDA topic modeling
- Visualize regional differences in framing and language use


## Expected Contributions

This study contributes to computational social science by:
- Demonstrating how linguistic framing can be operationalized and quantified.
- Highlighting ideological differences in political discourse.
- Providing a reproducible, transparent pipeline for analyzing contested political narratives.
- Providing a comparison of 2 different NLP models for sentiment analysis on multilingual datasets.

## Project Structure
```
css_news_sentiment/
├── 1_collect_data.py         # Data collection
├── 2_analyze_sentiment.py    # Sentiment analysis (BERT, XLM)
├── 3_visualize.py            # Charts and heatmaps
├── 4_wordclouds.py           # Word cloud generation
├── articles.csv              # Raw collected articles
├── articles by region/       # CSVs split by region
├── results-sentiment-BERT/   # BERT sentiment results
├── results-sentiment-XLM/    # XLM sentiment results
├── results-emotion-NRC-bigram-LDA/  # NRC, bigram & LDA results
├── results-wordcloud/        # Word cloud images
├── nrc_lexicon.txt           # NRC Emotion Lexicon
└── requirements.txt
```

## Tools & Libraries

| Tool | Purpose |
|---|---|
| `transformers` (HuggingFace) | BERT & XLM-RoBERTa sentiment models |
| `nltk` | Text preprocessing |
| `nrc_lexicon.txt` | Emotion lexicon (NRC) |
| `gensim` | LDA topic modeling |
| `matplotlib` / `seaborn` | Visualization |
| `wordcloud` | Word cloud generation |
| `pandas` | Data handling |

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/lorebueno/css-wise-2526.git
cd css-wise-2526
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run each step individually:
```bash
python 1_collect_data.py
python 2_analyze_sentiment.py
python 3_visualize.py
python 4_wordclouds.py
```

## Author

**Lore Bueno** · Matrikel-Nummer: 12835250
Ludwig-Maximilians-Universität - LMU
Statistics and Data Science Master's Program
Computational Social Science  - WISE 25/26
GitHub: [@lorebueno](https://github.com/lorebueno)