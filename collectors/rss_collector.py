"""
MarketPulse — Collecteur RSS (Couche Bronze)
Collecte les articles depuis les flux RSS et les stocke bruts en JSON.
"""

import feedparser
import json
import hashlib
import os
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
from bs4 import BeautifulSoup
import yaml
import logging

# Configuration logging pour Airflow
logger = logging.getLogger("airflow.task")

# Configuration globale via variables d'environnement
BRONZE_PATH = Path("/opt/airflow/data/bronze")
BRONZE_PATH.mkdir(parents=True, exist_ok=True)
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES_PER_SOURCE", 100))
CONFIG_PATH = "/opt/airflow/dags/collectors/sources.yaml"

def load_sources(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Charge la configuration des sources RSS."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_article_id(url: str, title: str) -> str:
    """Génère un ID unique basé sur l'URL et le titre."""
    content = f"{url}{title}".encode("utf-8")
    return hashlib.md5(content).hexdigest()


def clean_html(html_text: str) -> str:
    """Nettoie le HTML et retourne du texte brut."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "lxml")
    return soup.get_text(separator=" ", strip=True)


def parse_date(entry) -> str:
    """Parse la date de publication depuis un entry feedparser."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def collect_from_source(source_config: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Collecte uniquement les métadonnées depuis une source RSS.
    Architecture orientée données brutes (Bronze).
    """
    articles = []

    name = source_config.get("name", "unknown")
    url = source_config.get("url")
    category = source_config.get("category", "general")   
    
    if not url:
        return []

    # 1. Headers pour simuler un vrai navigateur et éviter le blocage 403
    Headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    logger.info(f"Collecte RSS depuis {name}: {url}")

    try:
        # 2. Utilisation de 'requests' pour gérer le timeout et les headers avant le parseur
        response = requests.get(url, headers=Headers, timeout=15)
        response.raise_for_status() # Lève une exception si erreur HTTP (404, 500...)
        feed = feedparser.parse(response.content)
        # Remplacement des prints par du logging DE propre
        logger.debug(f"Nombre d'entrées détectées pour {name} : {len(feed.entries)}")
        if feed.entries:
            logger.debug(f"Clés du premier article ({name}) : {list(feed.entries[0].keys())}")
        if feed.bozo:
            logger.warning(f"Flux malformé pour {name}: {feed.bozo_exception}")
            

        for entry in feed.entries[:MAX_ARTICLES]:
            title = clean_html(getattr(entry, "title", ""))
            summary = clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            link = getattr(entry, "link", "")

            # Validation minimale : un article sans titre ou lien est inutile
            if not title or not link:
                continue

            # Construction de l'objet article (Bronze)
            article = {
                "id": generate_article_id(link, title),
                "title": title,
                "summary": summary,
                "url": link,
                "source_name": name,
                "category": category,
                "published_at": parse_date(entry),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                # On initialise full_content à vide, l'étape Silver s'en chargera
                "full_content": "" 
            }

            articles.append(article)

        logger.info(f"Source {name}: {len(articles)} articles collectés.")
    except Exception as e:
        logger.error(f"Échec collecte {name}: {e}")
    return articles

def save_bronze(articles: List[dict], run_id: str) -> Path:
    """
    Sauvegarde les articles bruts dans la couche Bronze (JSON partitionné par date).

    La donnée est encapsulée dans un dictionnaire pour respecter le contrat de validation.
    """
    output_dir = BRONZE_PATH / datetime.now().strftime("%Y/%m/%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"all_articles.json"
    

    # # 3. Sauvegarde atomique avec gestion d'erreur robuste
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "count": len(articles),
                    "collected_at": datetime.now(timezone.utc).isoformat()
                },
                "articles": articles
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Sauvegarde centralisée réussie : {output_file}")
        return output_file

    except IOError as e:
        logger.error(f"Erreur d'écriture disque pour {output_file}: {e}")
        raise # # On relève l'erreur pour que Airflow marque la tâche en 'Failed'

def run_collection(config_path: str = CONFIG_PATH,
                   run_id: str = None) -> Dict[str, Any]:
    """
    Point d'entrée principal — collecte depuis toutes les sources.

    Returns:
        Dictionnaire avec stats et chemin de sortie
    """
    sources_config = load_sources(config_path)
    all_articles = []
    seen_ids = set()
    sources_dict = sources_config.get("sources", {})
        # 2. Collecte sécurisée
    if isinstance(sources_dict, dict):
        for category, sources in sources_dict.items():
            for source in sources:
                source["category"] = category
                for art in collect_from_source(source):
                    if art["id"] not in seen_ids:
                        seen_ids.add(art["id"])
                        all_articles.append(art)
    elif isinstance(sources_dict, list):
        for source in sources_dict:
            for art in collect_from_source(source):
                if art["id"] not in seen_ids:
                    seen_ids.add(art["id"])
                    all_articles.append(art)

    if not all_articles:
        raise ValueError("Aucun article collecté : le pipeline doit s'arrêter.")

        # 3. Sauvegarde centralisée vers le dossier Bronze
    output_file_path = save_bronze(all_articles, run_id)

    logger.info(f"Collecte terminée. Total: {len(all_articles)} articles.")

    return {
        "output_file": str(output_file_path),
        "metadata": {"total": len(all_articles)},
        "status": "success"
    }

# --- POUR LE TEST LOCAL UNIQUEMENT ---


if __name__ == "__main__":
    run_collection()
