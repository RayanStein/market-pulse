"""
MarketPulse — Pipeline NLP (Couche Silver)
Nettoyage, normalisation, embeddings SBERT multilingue.
"""

import json
import os
import re
import string
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer


SILVER_PATH = Path(os.getenv("SILVER_PATH", "./data/silver"))
BRONZE_PATH = Path(os.getenv("BRONZE_PATH", "./data/bronze"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# Modèle chargé une seule fois (singleton)
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Chargement du modèle SBERT: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ─── Nettoyage texte ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Nettoie un texte brut pour le NLP."""
    if not text:
        return ""
    # Supprime URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Supprime HTML résiduel
    text = re.sub(r"<[^>]+>", "", text)
    # Normalise espaces
    text = re.sub(r"\s+", " ", text).strip()
    # Supprime caractères de contrôle
    text = "".join(c for c in text if c.isprintable())
    return text


def build_document(article: dict) -> str:
    """
    Construit un document texte enrichi depuis un article.
    Combine titre + résumé avec pondération.
    """
    title = clean_text(article.get("title", ""))
    summary = clean_text(article.get("summary", ""))
    # Titre x3 pour donner plus de poids sémantique
    return f"{title}. {title}. {title}. {summary}"


def filter_article(article: dict, min_length: int = 50) -> bool:
    """Filtre les articles trop courts ou sans contenu utile."""
    title = article.get("title", "")
    summary = article.get("summary", "")
    combined = f"{title} {summary}"
    return len(combined.strip()) >= min_length


# ─── Enrichissement ─────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 10, lang: str = "fr") -> List[str]:
    """Extrait les mots-clés en supportant l'arabe et le latin."""
    # 1. Stop words étendus (Français + Anglais + Arabe de base)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "has", "have", "had", "this", "that", "it", "its",
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "en",
        "que", "qui", "est", "dans", "par", "sur", "au", "aux", "ce","في","هي","كما," "من", "على", "عن", "الى", "التي", "الذي", "هذا", "كان"
    }
    # 2. Expression régulière conditionnelle
    # [\u0600-\u06FF] capture l'alphabet arabe
    # [a-zA-ZÀ-ÿ] capture l'alphabet latin accentué
    regex = r"[\u0600-\u06FF]{3,}|[a-zA-ZÀ-ÿ]{4,}"
    words = re.findall(regex, text.lower())
    freq = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def compute_text_stats(text: str) -> dict:
    """Calcule des statistiques basiques sur le texte."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": (
            round(sum(len(w) for w in words) / len(words), 2)
            if words else 0
        ),
        "char_count": len(text),
    }


# ─── Pipeline principal ──────────────────────────────────────────────────────

def process_bronze_file(bronze_file: Path, run_id: str) -> Path:
    """
    Traite un fichier Bronze → produit un fichier Silver enrichi avec embeddings.

    Args:
        bronze_file: Chemin vers le fichier JSON Bronze
        run_id: ID du run

    Returns:
        Chemin vers le fichier Silver produit
    """
    logger.info(f"Traitement NLP: {bronze_file}")

    with open(bronze_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    # Robustesse professionnelle : extraction sécurisée des articles selon le format du JSON
    if isinstance(content, dict):
        articles = content.get("articles", content.get("data", []))
    elif isinstance(content, list):
        articles = content
    else:
        articles = []

    model = get_model()
    processed = []
    documents = []

    for art in articles:
        if not isinstance(art, dict):
            continue
        if not filter_article(art):
            continue
        doc = build_document(art)
        documents.append(doc)
        processed.append(art)

    if not documents:
        logger.warning("Aucun article valide à traiter")
        return None

    # Calcul des embeddings en batch (efficace)
    logger.info(f"Calcul de {len(documents)} embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Enrichissement des articles
    silver_articles = []
    for i, art in enumerate(processed):
        enriched = {
            **art,
            "url": art.get("url", ""),
            "cleaned_text": clean_text(
                f"{art.get('title','')} {art.get('summary','')}"
            ),
            "document": documents[i],
            "embedding": embeddings[i].tolist(),
            "keywords": extract_keywords(documents[i], lang=art.get("language", "fr")),
            "text_stats": compute_text_stats(documents[i]),
            "processed_at": datetime.utcnow().isoformat(),
        }
        silver_articles.append(enriched)

    # Sauvegarde Silver
    date_str = datetime.now().strftime("%Y/%m/%d")
    output_dir = SILVER_PATH / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"silver_{run_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(silver_articles, f, ensure_ascii=False, indent=2)

    logger.success(
        f"Silver: {len(silver_articles)} articles enrichis → {output_file}"
    )
    return output_file


if __name__ == "__main__":
    # Test rapide sur un fichier bronze existant
    import sys
    if len(sys.argv) > 1:
        result = process_bronze_file(Path(sys.argv[1]), "test_run")
        print(f"Fichier Silver produit: {result}")
