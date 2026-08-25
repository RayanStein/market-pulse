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
from nlp.utils import MULTILINGUAL_STOPWORDS

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
                    min_cluster_size: int = 8,min_samples: int = 2) -> np.ndarray:
    """
    Clustering HDBSCAN — détecte automatiquement le nombre de clusters.
    Idéal pour des corpus de taille variable.
    Les articles avec label=-1 sont des outliers (non assignés).

    Clustering HDBSCAN avec normalisation L2 préalable pour forcer 
    un comportement cosinus tout en gardant 'euclidean'.
    """

    # Normalisation L2 : indispensable pour que la distance euclidienne réagisse comme une similarité cosinus sur des embeddings SBERT.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Évite la division par zéro
    embeddings_normalized = embeddings / norms

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings_normalized)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(embeddings)
    # Garde-fou MLOps : si HDBSCAN est inopérant sur ce batch, fallback propre sur KMeans
    if n_clusters < 3 or noise_ratio > 0.40:
        logger.warning(
            f"⚠️ HDBSCAN inopérant ({n_clusters} clusters, {n_noise} bruits [{noise_ratio:.1%}]). "
            "Basculement automatique sur KMeans optimal."
        )
        kmeans_labels, silhouette_val = cluster_kmeans_optimal(embeddings_normalized)
        return kmeans_labels, silhouette_val

    # Calcul de la silhouette pour HDBSCAN si valide
    try:
        silhouette_val = float(silhouette_score(embeddings_normalized, labels, sample_size=min(500, len(embeddings))))
    except Exception:
        silhouette_val = 0.0

    logger.info(f"HDBSCAN Validé (Normalisé L2): {n_clusters} clusters, {n_noise} outliers sur {len(embeddings)} articles")
    return labels, silhouette_val

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
def label_cluster(articles_in_cluster: list[dict]) -> str:
    """
    Génère automatiquement un label pour un cluster en extrayant 
    les termes les plus pertinents via CountVectorizer (bigrammes inclus) 
    tout en filtrant les stopwords multilingues avec un fallback garanti sur des mots-clés 
    et non sur un titre brut.
    """
    texts = [art.get("title", "") for art in articles_in_cluster if art.get("title")]
    if not texts:
        return "divers · general"

    try:
        # Configuration d'un vectoriseur textuel robuste sur les titres
        vectorizer = CountVectorizer(
            stop_words=list(MULTILINGUAL_STOPWORDS),
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

        if top_words:
            return " · ".join(top_words)


    except Exception as e:
        logger.warning(f"Erreur lors du labeling automatique: {e}. Fallback mots-clés.")

    # Fallback propre orienté mots-clés (jamais de phrase brute)
    words = [
        w.lower() for w in texts[0].split()
        if len(w) > 3 and w.lower() not in MULTILINGUAL_STOPWORDS
    ]
    fallback_words = words[:3] if words else ["cluster", "analyse"]
    return " · ".join(fallback_words)

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
    if HDBSCAN_AVAILABLE and len(articles) >= 10:
        labels, silhouette_val = cluster_hdbscan(embeddings)
        algo_name = "hdbscan_with_fallback"
        # Calcul optionnel de la silhouette sur HDBSCAN pour les métriques MLflow
        try:
            if len(set(labels)) > 1:
                silhouette_val = float(silhouette_score(embeddings, labels, sample_size=min(500, len(embeddings))))
        except Exception:
            silhouette_val = 0.0
    else:
        labels, silhouette_val = cluster_kmeans_optimal(embeddings)
        algo_name = "kmeans"
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
        mlflow.set_experiment("marketpulse_prod")
        # SÉCURITÉ MLOPS : Fermeture forcée de tout run resté fantôme en mémoire
        if mlflow.active_run():
            logger.warning("⚠️ Run MLflow orphelin détecté. Fermeture forcée avant initialisation.")
            mlflow.end_run()

        with mlflow.start_run(run_name=f"clustering_{run_id}"):
            mlflow.log_param("algorithm",
                             "hdbscan" if HDBSCAN_AVAILABLE and len(articles) >= 10 else algo_name)
            mlflow.log_param("n_articles", len(articles))
            mlflow.log_metric("n_clusters", n_clusters)
            mlflow.log_metric("n_outliers", n_outliers)
            mlflow.log_metric(
                "outlier_rate", round(n_outliers / len(articles), 3)
            )
            mlflow.log_metric("silhouette_score", silhouette_val)
            logger.success("✅ Métriques de traçabilité MLOps enregistrées dans MLflow avec succès.")

    except Exception as e:
        logger.warning(f"⚠️  Avertissement MLflow : Impossible de joindre le serveur de tracking ({e}). Poursuite du pipeline en mode dégradé.")

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
