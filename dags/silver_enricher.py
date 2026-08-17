import json

def enrich_articles(articles):
    # Logique d'enrichissement (Silver)
    for article in articles:
        # Exemple d'enrichissement simple
        article["enriched"] = True
        article["status"] = "silver"
    return articles
