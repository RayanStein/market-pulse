"""
MarketPulse — Clustering thématique (HDBSCAN + labeling automatique)
Regroupe les articles par thèmes sémantiques proches.
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional
from loguru import logger

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN non disponible, fallback sur KMeans")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import mlflow


GOLD_PATH = Path(os.getenv("GOLD_PATH", "./data/gold"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://marketpulse_mlflow_v2:5000")
N_MIN = int(os.getenv("N_CLUSTERS_MIN", 3))
N_MAX = int(os.getenv("N_CLUSTERS_MAX", 15))


# ─── Clustering ─────────────────────────────────────────────────────────────

def cluster_hdbscan(embeddings: np.ndarray,
                    min_cluster_size: int = 3) -> np.ndarray:
    """
    Clustering HDBSCAN — détecte automatiquement le nombre de clusters.
    Idéal pour des corpus de taille variable.
    Les articles avec label=-1 sont des outliers (non assignés).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="cosine",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(
        f"HDBSCAN: {n_clusters} clusters, {n_noise} outliers"
        f" sur {len(embeddings)} articles"
    )
    return labels


def cluster_kmeans_optimal(embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    """
    KMeans avec recherche du nombre optimal de clusters via score silhouette.
    Fallback si HDBSCAN non disponible ou corpus trop petit.
    """
    n_samples = len(embeddings)
    k_max = min(N_MAX, n_samples - 1)
    k_min = min(N_MIN, k_max)

    best_k, best_score, best_labels = k_min, -1, None

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels, sample_size=min(500, n_samples))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    if best_labels is None:
        # Fallback de sécurité si aucun k valide
        best_labels = np.zeros(n_samples, dtype=int)
        best_score = 0.0

    logger.info(f"KMeans optimal: k={best_k}, silhouette={best_score:.3f}")
    return best_labels, float(best_score)


def reduce_for_viz(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Réduit les embeddings en 2D pour visualisation (Power BI scatter)."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


# ─── Labeling automatique ────────────────────────────────────────────────────
# Stopwords combinés (Français, Anglais, Allemand + bruits courants du web)
MULTILINGUAL_STOPWORDS = [
    # Anglais
    "the", "a", "an", "in", "on", "at", "for", "to", "with", "by", "is", "are", 
    "was", "were", "and", "or", "but", "of", "from", "its", "it", "as", "that", 
    "this", "they", "will", "says", "said", "can", "has", "have", "not",

    # Français
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "à", "au", 
    "aux", "que", "qui", "ce", "cette", "ces", "dans", "sur", "par", "pour", 
    "pas", "plus", "sont", "est", "ou", "mais", "ont", "fait",

    # Allemand (Complété avec pronoms et auxiliaires)
    "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "eines", "und", 
    "in", "im", "von", "zu", "den", "mit", "sich", "auf", "für", "ist", "nicht", 
    "nach", "wie", "als", "auch", "es", "an", "werden", "aus", "außer",
    "ich", "du", "er", "sie", "wir", "ihr", "mich", "dich", "ihn", "uns", "euch", "ihnen",
    "dem", "des", "über", "unter", "durch", "gegen", "ohne", "um", "oder", "aber", 
    "denn", "weil", "dass", "wenn", "so", "nur", "noch", "schon", "sind", "war", 
    "waren", "sein", "wird", "wurde", "wurden", "haben", "hat", "hatte", "hatten",

    # Bruit d'actualités / Métadonnées Scraping
    "promo", "off", "august", "auguste", "save", "deals", "code", "codes", "discount",

    # Bruit Temporel (Jours, mois, saisons)
    "jour", "jours", "day", "days", "tag", "tage", "يوم", "أيام",
    "mois", "month", "months", "monat", "monate", "شهر", "شهور", "أشهر",
    "année", "year", "years", "jahr", "jahre", "سنة", "سنوات", "عام", "أعوام",
    "saison", "season", "seasons", "موسم", "مواسم",
    "aujourd", "hui", "today", "heute", "demain", "hier", "اليوم", "غدا", "أمس",

    # Superlatifs & Qualificatifs génériques
    "good", "best", "better", "bad", "worst", "great",
    "bon", "meilleur", "pire", "bien", "très",
    "gut", "besser", "beste", "schlecht",
    "جيد", "أفضل", "أحسن", "سيء", "أسوأ", "عظيم", "ممتاز", "جدا",

    # Bruit Marketing & E-commerce complémentaire
    "annonce", "annonces", "ad", "ads", "advertisement", "إعلان", "إعلانات",
    "gratuit", "free", "premium", "sponsor", "sponsored", "مجاني", "مجانا", "مميز", "ممول", "برعاية",
    "abonnement", "subscribe", "newsletter", "cliquez", "click", "اشتراك", "اشترك", "نشرة", "إخبارية", "انقر", "اضغط",

    # Arabe (mots vides fréquents, prépositions et conjonctions)
    "في", "من", "على", "أن", "إلى", "عن", "هذا", "هذه", "التي", "الذي", 
    "ففي", "ولكن", "مع", "هل", "قد", "بل", "لا", "ما", "لم", "لن", 
    "هو", "هي", "هم", "هن", "أنت", "أنتم", "نحن", "وإلى", "وكما", "أو"
]
def label_cluster(articles_in_cluster: list[dict]) -> str:
    """
    Génère automatiquement un label pour un cluster en extrayant 
    les termes les plus pertinents via CountVectorizer (bigrammes inclus) 
    tout en filtrant les stopwords multilingues.
    """
    texts = [art.get("title", "") for art in articles_in_cluster if art.get("title")]
    if not texts:
        return "Thème divers"

    try:
        # Configuration d'un vectoriseur textuel robuste sur les titres
        vectorizer = CountVectorizer(
            stop_words=MULTILINGUAL_STOPWORDS,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b([a-zA-ZÀ-ÿ]{3,}|[\u0600-\u06FF]{2,})\b' 
            # Note : min 3 lettres pour le latin, min 2 caractères pour l'arabe (les mots en arabe sont souvent plus courts morphologiquement)
        )
        
        X = vectorizer.fit_transform(texts)
        sum_words = X.sum(axis=0)
        
        words_freq = [
            (word, sum_words[0, idx]) 
            for word, idx in vectorizer.vocabulary_.items()
        ]
        
        # Trier par fréquence décroissante
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        # Conserver le top 3 des mots/bigrammes les plus discriminants
        top_words = [w[0] for w in words_freq[:3]]
        
        if not top_words:
            return texts[0][:40] + "..."
            
        return " · ".join(top_words)
        
    except Exception as e:
        logger.warning(f"Erreur lors du labeling automatique: {e}. Fallback titre.")
        return texts[0][:40] + "..."

def assign_clusters(articles: list[dict],
                    labels: np.ndarray) -> tuple[list[dict], dict]:
    """Assigne les clusters aux articles et génère les labels."""
    # Grouper les articles par cluster
    clusters = {}
    for i, art in enumerate(articles):
        label = int(labels[i])
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(art)

    # Générer labels pour chaque cluster
    cluster_labels = {}
    for cluster_id, cluster_arts in clusters.items():
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Non classifié"
        else:
            cluster_labels[cluster_id] = label_cluster(cluster_arts)

    # Enrichir les articles
    enriched = []
    for i, art in enumerate(articles):
        cluster_id = int(labels[i])
        enriched.append({
            **art,
            "cluster_id": cluster_id,
            "cluster_label": cluster_labels[cluster_id],
            "is_outlier": cluster_id == -1,
        })

    return enriched, cluster_labels


# ─── Pipeline principal ──────────────────────────────────────────────────────

def run_clustering(silver_file: Path, run_id: str) -> Path:
    """
    Pipeline complet: charge Silver → cluster → sauvegarde Gold.
    Trace les métriques dans MLflow.

    Args:
        silver_file: Chemin fichier Silver avec embeddings
        run_id: ID du run Airflow

    Returns:
        Chemin vers le fichier Gold
    """
    logger.info(f"Clustering: {silver_file}")

    with open(silver_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if len(articles) < 3:
        logger.warning("Trop peu d'articles pour clustérer")
        return None

    # Reconstruction matrice embeddings
    embeddings = np.array([art["embedding"] for art in articles])

    # Choix de l'algorithme avec score silhouette garanti
    silhouette_val = 0.0
    if HDBSCAN_AVAILABLE and len(articles) >= 10:
        labels = cluster_hdbscan(embeddings)
        # Calcul optionnel de la silhouette sur HDBSCAN pour les métriques MLflow
        try:
            if len(set(labels)) > 1:
                silhouette_val = float(silhouette_score(embeddings, labels, sample_size=min(500, len(embeddings))))
        except Exception:
            silhouette_val = 0.0
    else:
        labels, silhouette_val = cluster_kmeans_optimal(embeddings)
    # Réduction 2D pour visualisation
    coords_2d = reduce_for_viz(embeddings)

    # Ajout coordonnées aux articles
    for i, art in enumerate(articles):
        art["x_viz"] = float(coords_2d[i, 0])
        art["y_viz"] = float(coords_2d[i, 1])

    # Assignation des clusters
    enriched_articles, cluster_labels = assign_clusters(articles, labels)

    # Statistiques par cluster
    cluster_stats = {}
    for cluster_id, label in cluster_labels.items():
        cluster_arts = [a for a in enriched_articles
                        if a["cluster_id"] == cluster_id]
        cluster_stats[str(cluster_id)] = {
            "label": label,
            "count": len(cluster_arts),
            "sources": list(set(a["source_name"] for a in cluster_arts)),
            "languages": list(set(a.get("language", "unknown") for a in cluster_arts)),
            "sample_titles": [a["title"] for a in cluster_arts[:3]],
        }

    # Calcul métriques globales
    n_clusters = len([k for k in cluster_labels if k != -1])
    n_outliers = sum(1 for a in enriched_articles if a["is_outlier"])

    # MLflow tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(run_name=f"clustering_{run_id}"):
            mlflow.log_param("algorithm",
                             "hdbscan" if HDBSCAN_AVAILABLE and len(articles) >= 10 else "kmeans")
            mlflow.log_param("n_articles", len(articles))
            mlflow.log_metric("n_clusters", n_clusters)
            mlflow.log_metric("n_outliers", n_outliers)
            mlflow.log_metric(
                "outlier_rate", round(n_outliers / len(articles), 3)
            )
            mlflow.log_metric("silhouette_score", silhouette_val)

    except Exception as e:
        logger.warning(f"MLflow non disponible: {e}")

    # Construction Gold
    gold_data = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat(),
        "stats": {
            "total_articles": len(enriched_articles),
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
        },
        "clusters": cluster_stats,
        "articles": enriched_articles,
    }

    # Sauvegarde Gold (sans les embeddings — trop lourds pour Power BI)
    gold_lite = {
        **gold_data,
        "articles": [
            {k: v for k, v in art.items() if k != "embedding"}
            for art in enriched_articles
        ],
    }

    date_str = datetime.now().strftime("%Y/%m/%d")
    output_dir = GOLD_PATH / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gold_{run_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gold_lite, f, ensure_ascii=False, indent=2)

    logger.success(
        f"Gold: {n_clusters} clusters, "
        f"{len(enriched_articles)} articles → {output_file}"
    )
    return output_file
