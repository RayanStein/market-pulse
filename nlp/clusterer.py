"""
MarketPulse — Clustering thématique (HDBSCAN + labeling automatique)
Regroupe les articles par thèmes sémantiques proches.
"""

import json
import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from typing import Optional, Any
from loguru import logger
from nlp.utils import MULTILINGUAL_STOPWORDS

try:
    import umap  # <--- Indispensable pour la projection non linéaire des embeddings SBERT
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN non disponible, fallback sur KMeans")

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import mlflow.sklearn

GOLD_PATH = Path(os.getenv("GOLD_PATH", "./data/gold"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://marketpulse_mlflow_v2:5000")
N_MIN = int(os.getenv("N_CLUSTERS_MIN", 3))
N_MAX = int(os.getenv("N_CLUSTERS_MAX", 10))

# Pattern partagé pour garantir la cohérence de la tokenisation (multilingue latin + arabe)
SHARED_TOKEN_PATTERN = r'(?u)\b([a-zA-ZÀ-ÿ]{3,}|[\u0600-\u06FF]{2,})\b'

# ─── Clustering ─────────────────────────────────────────────────────────────

def cluster_hdbscan_dynamic(embeddings: np.ndarray) -> tuple[np.ndarray, Any, float, str]:
    """
    Clustering HDBSCAN adaptatif à l'échelle. 
    Les hyperparamètres évoluent de manière logarithmique et non linéaire avec N.
    """
    n_samples = len(embeddings)

    # 1. Normalisation L2 stricte
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)

    # 2. Paramétrage mathématique dynamique
    # min_cluster_size évolue doucement (racine carrée) pour éviter l'explosion à haute échelle
    dynamic_min_cluster = max(10, int(math.sqrt(n_samples)))
    dynamic_n_neighbors = min(50, max(5, int(math.sqrt(n_samples))))

    logger.info(f"UMAP: n_neighbors={dynamic_n_neighbors} | HDBSCAN: min_cluster_size={dynamic_min_cluster}")

    try:
        # Étape structurelle : UMAP compresse l'espace latent en préservant la topologie sémantique
        reducer = umap.UMAP(
            n_components=5,  # Sécurité dimensionnelle
            n_neighbors=dynamic_n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )
        embeddings_reduced = reducer.fit_transform(embeddings_normalized)
    except Exception as e:
        logger.error(f"UMAP failure: {e}. Bascule sur KMeans.")
        return cluster_kmeans_adaptive(embeddings_normalized)

    # 🟡 [OPTIMISATION 2] : Adaptation dynamique de min_cluster_size si le batch est très restreint,
    # pour éviter qu'un lot réduit ne soit entièrement rejeté en tant que bruit (-1).

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=dynamic_min_cluster,
        min_samples=max(3, dynamic_min_cluster // 3),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings_reduced)

    # 3. Sécurité MLOps : Taux de bruit
    noise_ratio = list(labels).count(-1) / n_samples
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Garde-fou MLOps : si HDBSCAN est inopérant sur ce batch, fallback propre sur KMeans
    if n_clusters < 2 or noise_ratio > 0.40:  # Tolérance au bruit rabaissée à 40% max
        logger.warning(f"HDBSCAN bruyant (Noise: {noise_ratio:.1%}). Bascule sur KMeans adaptatif.")
        return cluster_kmeans_adaptive(embeddings_normalized)

    # 4. Calcul Silhouette propre (hors bruit)
    valid_mask = labels != -1
    silhouette_val = float(silhouette_score(
        embeddings_reduced[valid_mask], 
        labels[valid_mask], 
        sample_size=min(1000, np.sum(valid_mask))
    )) if np.sum(valid_mask) > 1 else 0.0

    return labels, clusterer, silhouette_val, "hdbscan_dynamic"

def cluster_kmeans_adaptive(embeddings: np.ndarray) -> tuple[np.ndarray, Any, float, str]:
    """
    Fallback KMeans utilisant MiniBatch pour la vitesse, avec K dynamique.
    """
    n_samples = len(embeddings)

    # K encadré dynamiquement : au moins 5, max 15 (ou racine de N/2)
    k_min = 5
    k_max = min(15, max(5, int(math.sqrt(n_samples / 2))))

    best_k, best_score, best_labels, best_model = k_min, -1, None, None

    # Recherche rapide du K optimal
    for k in range(k_min, k_max + 1):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(1000, n_samples))

        if score > best_score:
            best_score, best_k, best_labels, best_model = score, k, labels, kmeans

    return best_labels, best_model, float(best_score), "kmeans_adaptive"

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
            token_pattern=SHARED_TOKEN_PATTERN
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
    if not articles:
        raise ValueError(f"Le fichier Silver {silver_file} est vide.")

    embeddings = np.array([art["embedding"] for art in articles])

    # 1. Clustering
    if HDBSCAN_AVAILABLE and len(articles) >= 20:
        labels, trained_model, silhouette_val, algo_name = cluster_hdbscan_dynamic(embeddings)
    else:
        labels, trained_model, silhouette_val, algo_name = cluster_kmeans_adaptive(embeddings)

    # Réduction 2D pour visualisation
    coords_2d = reduce_for_viz(embeddings)

    # Ajout coordonnées aux articles
    for i, art in enumerate(articles):
        art["x_viz"] = float(coords_2d[i, 0])
        art["y_viz"] = float(coords_2d[i, 1])

    # Assignation des clusters
    enriched_articles, cluster_labels = assign_clusters(articles, labels)

    # Calcul métriques globales
    n_clusters = len([k for k in cluster_labels if k != -1])
    n_outliers = sum(1 for a in enriched_articles if a["is_outlier"])

    # On force la limitation des clusters envoyés à l'étape suivante pour protéger Ollama
    n_clusters_total = len([k for k in cluster_labels if k != -1])
    if n_clusters_total > 15:
        logger.warning(f"Sur-segmentation détectée ({n_clusters} clusters). Limite recommandée dépassée.")


    # Statistiques par cluster (C'est ce dictionnaire qui doit nourrir le LLM)
    cluster_stats = {}
    for cluster_id, label in cluster_labels.items():
        cluster_arts = [a for a in enriched_articles
                        if a["cluster_id"] == cluster_id]
        # Sécurité : on ne traite que les clusters contenant des articles

        if cluster_arts:
            cluster_stats[str(cluster_id)] = {
                "label": label,
                "count": len(cluster_arts),
                "sources": list(set(a["source_name"] for a in cluster_arts)),
                "languages": list(set(a.get("language", "unknown") for a in cluster_arts)),
                "sample_titles": [a["title"] for a in cluster_arts[:3]], # Top 3 pour limiter le payload
            }

    # MLflow tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("marketpulse_prod")
        # SÉCURITÉ MLOPS : Fermeture forcée de tout run resté fantôme en mémoire
        if mlflow.active_run():
            logger.warning("⚠️ Run MLflow orphelin détecté. Fermeture forcée avant initialisation.")
            mlflow.end_run()

        with mlflow.start_run(run_name=f"clustering_{run_id}"):
            # 1. Tags de gouvernance (Standard MLOps pour filtrage dans l'UI)
            mlflow.set_tags({
                "pipeline": "marketpulse_clustering",
                "env": "production",
                "model_type": algo_name
            })

            # 2. Batch logging des paramètres (Optimisation réseau)
            mlflow.log_params({
                "algorithm": algo_name,
                "n_articles": len(articles)
            })

            # 3. Batch logging des métriques (Cast en float pour sécurité de sérialisation JSON)
            mlflow.log_metrics({
                "n_clusters": int(n_clusters),
                "n_outliers": int(n_outliers),
                "outlier_rate": float(round(n_outliers / len(articles), 3)),
                "silhouette_score": float(silhouette_val)
            })

            # ─── AJOUT MODEL REGISTRY MLOPS ───# Enregistrement sécurisé du modèle entraîné (Pipeline complet UMAP + HDBSCAN ou KMeans)
            # Remplacement du dump pickle manuel par l'API Sklearn native

            if trained_model is not None:
                logger.info(f"📦 Enregistrement du modèle natif MLflow : {algo_name}")
                # HDBSCAN et K-Means respectent tous deux l'API Scikit-Learn
                mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    artifact_path="model_registry",
                    registered_model_name=f"MarketPulse_Clustering_{algo_name.capitalize()}"
                )
                logger.success("✅ Modèle et environnement versionnés dans le Model Registry MLflow.")
            else:
                logger.warning("⚠️ trained_model est None, aucun modèle enregistré.")

    except Exception as e:
        import traceback
        logger.error(f"⚠️ Avertissement MLOps : Échec du tracking MLflow ({str(e)}). Le pipeline continue...")
        logger.debug(traceback.format_exc())
    # Construction Gold
    gold_data = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_articles": len(enriched_articles),
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
        },
        "clusters": cluster_stats,
        "articles": [
            {k: v for k, v in art.items() if k != "embedding"}
            for art in enriched_articles
        ],
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
