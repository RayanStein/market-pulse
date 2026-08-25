"""
MarketPulse — Générateur de synthèses par LLM (ollama)
Génère un rapport rédigé par cluster thématique.
"""

import os
import json
import mlflow
import asyncio
import httpx
from typing import List
from pathlib import Path
from datetime import datetime
from loguru import logger
from ollama import Client


# Initialisation légère (pas d'appel réseau synchrone au parsing)
ollama_client = Client(host='http://ollama:11434')
GOLD_PATH = Path(os.getenv("GOLD_PATH", "./data/gold"))
_ollama_semaphore = asyncio.Semaphore(3)

def build_cluster_prompt(cluster_label: str,
                         articles: List[dict],
                         language: str = "fr") -> str:
    """Construit le prompt pour la synthèse d'un cluster."""
    articles_text = "\n".join([
        f"- [{art['source_name']}] {art['title']}: {art['summary'][:200]}"
        for art in articles[:10]  # Max 10 articles par cluster
    ])

    if language == "fr":
        return f"""Tu es un journaliste data expert. Voici {len(articles)} articles regroupés sous le thème : "{cluster_label}".

Articles :
{articles_text}

Rédige une synthèse professionnelle de 3-4 phrases qui :
1. Résume les tendances principales
2. Identifie les points clés communs
3. Termine par une implication ou perspective

Réponds uniquement avec la synthèse, sans introduction ni titre."""
    else:
        return f"""You are a data journalist. Here are {len(articles)} articles grouped under the theme: "{cluster_label}".

Articles:
{articles_text}

Write a 3-4 sentence professional summary that:
1. Summarizes the main trends
2. Identifies key common points
3. Ends with an implication or outlook

Respond only with the summary, no introduction or title."""


def generate_cluster_synthesis(cluster_label: str, articles: list, lang: str = "fr") -> str:
    """Génère une synthèse pour un cluster via le LLM Llama 3 local."""
    
    # 1. On prépare la liste des articles textuels pour le prompt
    articles_text = "\n".join([
        f"- [{art.get('source_name', 'Source')}] {art.get('title', 'Sans titre')}: {art.get('summary', '')[:200]}"
        for art in articles[:10]  # Max 10 articles pour ne pas surcharger
    ])
    
    # 2. Construction du prompt multilingue ciblé
    prompt = f"""
    Tu es un expert en veille économique et stratégique. 
    Voici une liste de titres et résumés d'articles de presse qui appartiennent au même groupe thématique '{cluster_label}' :
    {articles_text}
    
    Fais un résumé global très court (maximum 2 phrases) en français pour expliquer ce qu'il se passe dans ce cluster.
    Sois direct,factuel, commence directement par le résumé, sans formules de politesse ni phrases d'introduction.
    """
    
    try:
        
        # 3. Appel à l'instance Ollama avec l'import global       
        response = ollama_client.chat(           
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.2} 
        )

        try:
            mlflow.log_param("llm_model", "llama3")
            # Récupération sécurisée des métriques d'Ollama
            if isinstance(response, dict):
                if 'eval_count' in response:
                    mlflow.log_metric("eval_count", response['eval_count'])
                if 'prompt_eval_count' in response:
                    mlflow.log_metric("prompt_eval_count", response['prompt_eval_count'])
        except Exception:
            pass
        return response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Erreur LLM Local pour le cluster {cluster_label}: {e}")
        return f"[Synthèse indisponible pour le thème '{cluster_label}']. Erreur: {str(e)}"



def generate_global_report(clusters_data, total_articles):
    """Génère un rapport global sur l'ensemble des clusters via Llama 3."""
    
    themes_summary = "\n".join([
        f"- {data['label']} ({data['count']} articles)"
        for cid, data in clusters_data.items()
        if cid != "-1"
    ])

    prompt = f"""Tu es un analyste de veille stratégique. Voici les {len(clusters_data)} thèmes principaux de la journée :
{themes_summary}

Rédige en 2-3 phrases un résumé exécutif de la veille du jour :
- Quel est le sujet dominant ?
- Quelles tendances émergent ?
- Quel est le niveau d'activité informationnelle ?

Commence par "Cette veille révèle..." ou similaire."""

    try:
        # Initialisation du client
        client = Client(host='http://ollama:11434')

        # Appel à ton Ollama local pour le rapport global
        response = client.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )


        try:
            mlflow.log_param("global_report_model", "llama3")
        except Exception:
            pass
        return response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Erreur génération rapport global: {e}")
        # En cas d'erreur, on renvoie un rapport par défaut élégant
        return f"Rapport MarketPulse - {total_articles} articles analysés, {len(clusters_data)} thèmes identifiés."

async def generate_single_cluster_with_semaphore(client: httpx.AsyncClient, cluster_id: str, cluster_info: dict, cluster_articles: list, lang: str) -> tuple:
    """Version asynchrone d'un appel individuel à Ollama pour un cluster."""
    cluster_label = cluster_info["label"]
    
    articles_text = "\n".join([
        f"- [{art.get('source_name', 'Source')}] {art.get('title', 'Sans titre')}: {art.get('summary', '')[:200]}"
        for art in cluster_articles[:10]
    ])
    
    prompt = f"""
    Tu es un expert en veille économique et stratégique. 
    Voici une liste de titres et résumés d'articles de presse qui appartiennent au même groupe thématique '{cluster_label}' :
    {articles_text}

    Fais un résumé global très court (maximum 2 phrases) en français pour expliquer ce qu'il se passe dans ce cluster.
    Sois direct, factuel, commence directement par le résumé, sans formules de politesse ni phrases d'introduction.
    """

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "options": {"temperature": 0.2},
        "stream": False
    }

    synthesis = None
    max_retries = 3
    backoff_factor = 2.0  # Délai exponentiel entre les tentatives (2s, 4s...)



    for attempt in range(1, max_retries + 1):
        try:
            # Le sémaphore protège l'appel critique pour ne pas saturer le serveur Ollama
            async with _ollama_semaphore:
                # Appel HTTP asynchrone direct sur l'API d'Ollama
                response = await client.post("http://ollama:11434/api/generate", json=payload, timeout=120.0)
            if response.status_code == 200:
                res_json = response.json()
                synthesis = res_json.get("response", "").strip()
                if synthesis:
                    break  # Succès, on sort de la boucle de retry
            else:
                logger.warning(f"[Tentative {attempt}/{max_retries}] Erreur HTTP Ollama {response.status_code} pour le cluster {cluster_label}")
        except Exception as e:
            logger.warning(f"[Tentative {attempt}/{max_retries}] Timeout/Erreur technique pour {cluster_label}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(backoff_factor ** attempt)

    # Si après les retries le LLM n'a vraiment pas répondu, on trace l'état technique proprement
    if not synthesis:
        synthesis = "[Synthèse en cours de génération - File d'attente LLM saturée]"
        logger.error(f"Échec définitif d'inférence pour le cluster {cluster_label} après {max_retries} essais.")

    enriched_data = {
        **cluster_info,
        "synthesis": synthesis,
        "top_sources": cluster_info["sources"][:5],
    }
    return cluster_id, enriched_data

async def generate_all_clusters_async(clusters: dict, articles_by_cluster: dict, report_lang: str) -> dict:
    """Exécute tous les appels de synthèse des clusters en parallèle simultané."""
    valid_clusters = {cid: cinfo for cid, cinfo in clusters.items() if cid != "-1"}
    # Sécurité : si aucun cluster valide n'émerge, on intègre le groupe -1 pour sauver le rapport
    if not valid_clusters and "-1" in clusters:
        valid_clusters = {"-1": clusters["-1"]}

    async with httpx.AsyncClient() as client:
        tasks = [
            generate_single_cluster_with_semaphore(client, cid, cinfo, articles_by_cluster.get(cid, []), report_lang)
            for cid, cinfo in clusters.items() if cid != "-1"
        ]
        results = await asyncio.gather(*tasks)
        return {cid: data for cid, data in results}

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

    # === OUVERTURE DU RUN MLFLOW ICI ===
    # 1. Configuration sécurisée de l'expérience MLflow
    start_time = datetime.utcnow()
    # Générer les synthèses de tous les clusters en parallèle (ultra rapide)
    logger.info("Lancement des synthèses de clusters en parallèle...")
    enriched_clusters = asyncio.run(
        generate_all_clusters_async(clusters, articles_by_cluster, report_lang)
    )

    # Calcul sécurisé de la durée
    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.success("Toutes les synthèses de clusters ont été générées en parallèle !")

        # Rapport global
    global_summary = generate_global_report(
        enriched_clusters, gold_data["stats"]["total_articles"]
    )


    # Rapport final
    report = {
        "run_id": run_id,
        "report_title": "MarketPulse — Rapport de veille",
        "generated_at": datetime.utcnow().isoformat(),
        "next_update": "Dans 2 heures",
        "global_summary": global_summary,
        "stats": gold_data["stats"],
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

    # Sauvegarde rapport
    output_dir = GOLD_PATH / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"report_{run_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.success(f"Rapport final généré: {output_file}")

    # 5. Télémétrie MLflow (Non-bloquante)

    try:
        # Configuration locale à l'exécution de la tâche (pas en haut du fichier !)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://marketpulse_mlflow_v2:5000"))
        mlflow.set_experiment("marketpulse_prod")

        with mlflow.start_run(run_name=f"synthesize_{run_id}") as run:
            mlflow.set_tag("airflow_run_id", run_id)
            mlflow.log_param("pipeline_stage", "synthesize_clusters")
            mlflow.log_param("llm_model", "llama3")
            mlflow.log_metric("total_clusters_to_synthesize", len([c for c in clusters.keys() if c != "-1"]))
            mlflow.log_metric("async_synthesis_duration_seconds", duration)
            logger.info("📊 Métriques enregistrées avec succès dans MLflow.")
    except Exception as e:
        logger.error(f"⚠️ MLflow tracking server unreachable (503/Timeout): {e}. Poursuite du pipeline sans télémétrie.")

    return output_file
