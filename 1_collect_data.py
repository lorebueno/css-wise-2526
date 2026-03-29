import os
import time
from dotenv import load_dotenv
from datetime import date
import mediacloud.api as mc
import pandas as pd

# Load the API key from .env
load_dotenv()
API_KEY = os.getenv("MEDIACLOUD_API_KEY")

# Connect to MediaCloud
search = mc.SearchApi(API_KEY)

# --- Regional Collection IDs ---
REGIONS = {
    # "Europe":        [34412146, 34412409, 34412476],  # France, Germany, UK - National
    # "Latin_America":  [34412043, 34412257, 34412358, 34412427],   # Argentina, Brasil, Colombia, Mexico - National
     "US":            [34412234],          # United States - National
    # "Venezuela":     [34412387, 38380333],  # Venezuela - National and State

}

def collect_articles(query: str, start_date: str, end_date: str, max_articles: int = 50):
    """
    Fetches news articles from MediaCloud for each region and saves to CSV.
    query        : keyword(s) to search
    start_date   : format "YYYY-MM-DD"
    end_date     : format "YYYY-MM-DD"
    max_articles : max number of articles to collect per region
    """
    all_articles = []

    for region_name, collection_ids in REGIONS.items():
        print(f"\nFetching articles for region: {region_name}...")

        # Pause between regions to avoid rate limiting
        time.sleep(5)

        region_articles = []
        pagination_token = None

        try:
            while len(region_articles) < max_articles:
                page, pagination_token = search.story_list(
                    query,
                    start_date=date.fromisoformat(start_date),
                    end_date=date.fromisoformat(end_date),
                    collection_ids=collection_ids,
                    pagination_token=pagination_token
                )

                if not page:
                    break

                for story in page:
                    region_articles.append({
                        "region":       region_name,
                        "title":        story.get("title", ""),
                        "url":          story.get("url", ""),
                        "publish_date": story.get("publish_date", ""),
                        "source":       story.get("media_name", "")
                    })

                print(f"  → fetched {len(region_articles)} articles so far...")

                if pagination_token is None:
                    break

            all_articles.extend(region_articles[:max_articles])
            print(f"  ✅ {min(len(region_articles), max_articles)} articles collected for {region_name}")

        except Exception as e:
            print(f"  ⚠️ Skipping {region_name} due to error: {e}")
            continue

        # Save everything to one CSV
        df = pd.DataFrame(all_articles)
        df.to_csv("articles.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ Total: {len(df)} articles saved to articles.csv")
    return df

# Run it!
if __name__ == "__main__":
    collect_articles(
        query="(article_title: Maduro OR Venezuela) AND (article_title: Trump OR EUA OR USA OR US)",
        start_date="2026-01-02",
        end_date="2026-01-08",
        max_articles=250
    )