"""
MarketPulse — Générateur de synthèses par LLM (ollama)
Génère un rapport rédigé par cluster thématique.
"""

import os
import json
import time
import mlflow
import asyncio
import httpx
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
from ollama import Client


# Initialisation légère (pas d'appel réseau synchrone au parsing)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://marketpulse_mlflow_v2:5000")
GOLD_PATH = Path(os.getenv("GOLD_PATH", "./data/gold"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

_ollama_semaphore = asyncio.Semaphore(int(os.getenv("OLLAMA_CONCURRENCY", 3)))

def build_cluster_prompt(cluster_label: str,
                         articles: List[dict],max_articles: int = 10,
                         language: str = "fr") -> str:
    """Construit un prompt optimisé. 
    Pré-requis: Le clusterer DOIT fournir les articles triés par représentativité (centroïde).
    """
    # Optimisation de la limite de contexte : on favorise des titres complets plutôt que des résumés tronqués
    articles_text = "\n".join([
        f"- [{art.get('source_name', 'Source Inconnue')}] {art.get('title', 'Sans titre')}"
        for art in articles[:max_articles]
    ])

    if language == "fr":
        return f"""Tu es un journaliste data expert. Voici {len(articles)} articles regroupés sous le thème : "{cluster_label}".

Articles :
{articles_text}

Rédige une synthèse professionnelle exécutive ultra-concise de 3-4 phrases qui :
1. Résume dynamiquemnt les tendances principales
2. Identifie les points clés communs
3. Termine par une implication ou perspective

Réponds uniquement avec la synthèse, sans introduction ni titre, commence directement par le contenu."""
    else:
        return f"""You are a data journalist. Here are {len(articles)} articles grouped under the theme: "{cluster_label}".

Articles:
{articles_text}

Write a 3-4 sentence professional summary that:
1. Summarizes the main trends
2. Identifies key common points
3. Ends with an implication or outlook

Respond only with the summary, no introduction or title."""


async def generate_single_cluster_async(client: httpx.AsyncClient, cluster_id: str, cluster_info: dict, cluster_articles: list, lang: str) -> dict:
    """Génère une synthèse pour un cluster via le LLM Llama 3 local."""
    cluster_label = cluster_info["label"]
    prompt = build_cluster_prompt(cluster_label, cluster_articles, max_articles=10, language=lang)

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "options": {"temperature": 0.1, "num_predict": 150},
        "stream": False
    }

    synthesis = "[Échec - Timeout LLM]"
    metrics = {"eval_count": 0, "throughput_tok_s": 0.0, "latency_sec": 0.0}
    max_retries = 3
    backoff_factor = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            async with _ollama_semaphore:
                start_time = time.perf_counter()
                # 🟩 FIX: Utilisation de la variable d'environnement OLLAMA_HOST au lieu de l'URL hardcodée
                response = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
                response.raise_for_status()

            res_json = response.json()
            synthesis = res_json.get("response", "").strip()
            latency_sec = time.perf_counter() - start_time
            eval_count = res_json.get("eval_count", 0)
            eval_duration_ns = res_json.get("eval_duration", 1)
            
            metrics = {
                "eval_count": eval_count,
                "latency_sec": round(latency_sec, 2),
                "throughput_tok_s": round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0.0
            }
            logger.debug(f"Cluster '{cluster_label}' synthétisé: {metrics['latency_sec']}s | {metrics['throughput_tok_s']} tok/s")
            break

        except httpx.HTTPError as e:
            logger.warning(f"[Essai {attempt}/{max_retries}] Erreur HTTP pour '{cluster_label}': {e}")
            if attempt < max_retries:
                await asyncio.sleep(backoff_factor ** attempt)
        except Exception as e:
            logger.error(f"[Essai {attempt}/{max_retries}] Erreur inattendue pour '{cluster_label}': {e}")
            break

    return {
        "cluster_id": cluster_id,
        "enriched_data": {**cluster_info, "synthesis": synthesis, "top_sources": cluster_info.get("sources", [])[:5]},
        "metrics": metrics
    }

async def generate_global_report_async(client: httpx.AsyncClient, clusters_data: dict, total_articles: int) -> str:
    """Génération du rapport global en utilisant le même client HTTPX asynchrone."""
    themes_summary = "\n".join([
        f"- {data['label']} ({data['count']} articles)"
        for cid, data in clusters_data.items() if cid != "-1"
    ])

    prompt = f"""Tu es un analyste de veille stratégique. Voici les thèmes principaux de la journée :
{themes_summary}

Rédige en 2-3 phrases un résumé exécutif direct. Commence par 'Cette veille révèle...'."""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "options": {"temperature": 0.2, "num_predict": 200},
        "stream": False
    }

    try:
        response = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Erreur rapport global: {e}")
        return f"Rapport MarketPulse - {total_articles} articles analysés, {len(clusters_data)} thèmes identifiés."

def enrich_gold_with_syntheses(gold_file: Path, run_id: str) -> Path:
    """
    Charge le fichier Gold, génère les synthèses LLM et produit
    le rapport final enrichi.

    Args:
        gold_file: Chemin vers le fichier Gold
        run_id: ID du run

    Returns:
        Chemin vers le fichier rapport final
    """
    logger.info(f"Génération des synthèses LLM: {gold_file}")

    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    clusters = gold_data.get("clusters", {})
    articles = gold_data.get("articles", [])
    total_articles = gold_data.get("stats", {}).get("total_articles", len(articles))
    # Regrouper les articles par cluster
    articles_by_cluster = {}
    for art in articles:
        cid = str(art["cluster_id"])
        if cid not in articles_by_cluster:
            articles_by_cluster[cid] = []
        articles_by_cluster[cid].append(art)

    # Détecter la langue dominante
    languages = [art.get("language", "en") for art in articles]
    dominant_lang = max(set(languages), key=languages.count) if languages else "en"
    report_lang = "fr" if dominant_lang in ["fr", "ar"] else "en"

    # Définition de valid_clusters (indispensable pour la boucle de génération)
    valid_clusters = {cid: cinfo for cid, cinfo in clusters.items() if cid != "-1"}
    if not valid_clusters and "-1" in clusters:
        valid_clusters = {"-1": clusters["-1"]}


    output_dir = GOLD_PATH / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"report_{run_id}.json"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("marketpulse_prod")


    with mlflow.start_run(run_name=f"synthesize_{run_id}") as run:
        mlflow.set_tag("airflow_run_id", run_id)
        mlflow.log_params({"llm_model": LLM_MODEL, "target_language": report_lang})

        try:
            start_time = time.perf_counter()

            async def run_pipeline():
                async with httpx.AsyncClient() as client:
                    tasks = [
                        generate_single_cluster_async(client, cid, cinfo, articles_by_cluster.get(cid, []), report_lang)
                        for cid, cinfo in valid_clusters.items()
                    ]
                    cluster_results = await asyncio.gather(*tasks)

                    enriched_clusters = {res["cluster_id"]: res["enriched_data"] for res in cluster_results}

                    for i, res in enumerate(cluster_results):
                        cid = res["cluster_id"]
                        mlflow.log_metrics({
                            f"cluster_{cid}_latency_s": res["metrics"]["latency_sec"],
                            f"cluster_{cid}_throughput_tps": res["metrics"]["throughput_tok_s"]
                        }, step=i)
 
                    global_summary = await generate_global_report_async(client, enriched_clusters, total_articles)
                    return enriched_clusters, global_summary

            # Exécution de la boucle asynchrone
            enriched_clusters, global_summary = asyncio.run(run_pipeline())

            pipeline_duration = time.perf_counter() - start_time
            mlflow.log_metric("total_pipeline_duration_sec", round(pipeline_duration, 2))

            # Rapport final
            report = {
                "run_id": run_id,
                "report_title": "MarketPulse — Rapport de veille",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "next_update": "Dans 2 heures",
                "global_summary": global_summary,
                "stats": gold_data.get("stats", {}),
                "clusters": enriched_clusters,
                "articles": [
                    {
                        "id": a["id"],
                        "title": a["title"],
                        "url": a["url"],
                        "source_name": a["source_name"],
                        "published_at": a["published_at"],
                        "cluster_id": a["cluster_id"],
                        "cluster_label": a.get("cluster_label", "Unclassified"),
                        "language": a.get("language", "en"),
                        "category": a.get("category", "general"),
                        "keywords": a.get("keywords", [])[:5],
                        "x_viz": a.get("x_viz", 0),
                        "y_viz": a.get("y_viz", 0),
                    }
                    for a in articles if not a.get("is_outlier", False)
                ],
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.success(f"Rapport final généré: {output_file}")
            mlflow.log_artifact(str(output_file), "final_reports")
            logger.success(f"Pipeline terminé. Artefact tracé dans MLflow: {output_file}")

        except Exception as e:
            logger.critical(f"Crash critique du pipeline de synthèse: {e}")
            mlflow.set_tag("status", "FAILED")
            raise e

    return output_file
