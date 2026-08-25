"""
MarketPulse — DAG Airflow principal
Pipeline complet: Collecte → NLP → Clustering → Synthèse LLM → API
Planifié toutes les 2 heures.
"""
import sys
from pathlib import Path

# Trouve dynamiquement le dossier racine d'Airflow (/opt/airflow) et l'ajoute au PATH Python
ROOT_DIR = Path("/opt/airflow/dags")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# Ajoute le dossier parent au PYTHONPATH pour trouver le package marketpulse
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone
import os
import json
import logging
import traceback
import requests
from glob import glob
# 2. Imports de tes modules (maintenant que le sys.path est corrigé, cela fonctionnera)
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from data_validator import validate_bronze_data # <--- Ton validateur
from silver_enricher import enrich_articles # <--- Ta nouvelle tâche
from collectors.rss_collector import run_collection # Ton collecteur
from logger_utils import get_pipeline_logger
from airflow.exceptions import AirflowSkipException
from airflow.utils.email import send_email
from nlp.preprocessor import process_bronze_file
from nlp.clusterer import run_clustering
from nlp.synthesizer import enrich_gold_with_syntheses

# Import de l'opérateur Slack avec fallback sécurisé
try:
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    SLACK_OPERATOR_AVAILABLE = True
except ImportError:
    SLACK_OPERATOR_AVAILABLE = False

logger = get_pipeline_logger("marketpulse_pipeline")

# Configuration des variables d'alerte (Variables d'environnement ou valeurs par défaut)
ALERT_EMAIL = os.getenv("MARKETPULSE_ALERT_EMAIL", "rayen_rezgui@yahoo.com")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")  # URL Webhook d'alerte Slack/Teams



# =============================================================================
# HELPER: CALCUL DE LA FENÊTRE D'EXÉCUTION (EXECUTION SLA)
# =============================================================================

def calculate_execution_duration(context) -> tuple[float, str]:
    """
    Calcule dynamiquement la durée d'exécution globale (end_time - start_time).
    Retourne (durée en secondes, chaîne formatée '186s (~3m 06s)').
    """
    dag_run = context.get("dag_run")
    if not dag_run or not dag_run.start_date:
        return 0.0, "N/A"
    
    start_time = dag_run.start_date
    now = datetime.now(timezone.utc) if start_time.tzinfo else datetime.utcnow()
    
    duration_seconds = round((now - start_time).total_seconds(), 2)
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    
    formatted_str = f"{int(duration_seconds)}s ({minutes}m {seconds:02d}s)"
    return duration_seconds, formatted_str

# =============================================================================
# CALLBACK D'ÉCHEC GLOBAL (ON_FAILURE_CALLBACK)
# =============================================================================

def notify_on_failure(context):
    """
    Callback exécuté automatiquement lors du crash d'une tâche.
    Alerte immédiatement l'équipe Data via Email ET Slack/Teams.
    """
    ti = context.get("task_instance")
    task_id = ti.task_id if ti else "Inconnue"
    dag_id = context.get("dag").dag_id if context.get("dag") else "marketpulse_pipeline"
    run_id = context.get("run_id", "N/A")
    execution_date = context.get("execution_date", datetime.utcnow())

    # [CORRECTIF #1] Calcul du temps écoulé avant l'échec
    _, duration_str = calculate_execution_duration(context)


# Capture et formatage de la trace d'erreur Python
    exception = context.get("exception", "Erreur inconnue")
    if hasattr(exception, "__traceback__") and exception.__traceback__:
        formatted_traceback = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    else:
        formatted_traceback = str(exception)
        
    logger.error(f"❌ ECHEC DE LA TÂCHE [{task_id}] dans le DAG [{dag_id}] après {duration_str}.")

    # --- 1. Notification E-mail ---
    email_subject = f"🚨 [CRITICAL ALERT] Échec Tâche '{task_id}' — DAG {dag_id}"
    email_body = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ebacd1; border-radius: 8px; background-color: #fdf2f2;">
        <h2 style="color: #c53030; margin-top: 0;">🚨 Alerte d'Échec Pipeline MarketPulse</h2>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr><td style="padding: 8px; font-weight: bold; background-color: #feb2b2; width: 30%;">DAG ID:</td><td style="padding: 8px; background-color: #fff5f5;">{dag_id}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold; background-color: #feb2b2;">Tâche Plantée:</td><td style="padding: 8px; background-color: #fff5f5; color: #c53030; font-weight: bold;">{task_id}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold; background-color: #feb2b2;">Run ID:</td><td style="padding: 8px; background-color: #fff5f5;">{run_id}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold; background-color: #feb2b2;">Temps écoule avant échec:</td><td style="padding: 8px; background-color: #fff5f5;"><b>{duration_str}</b></td></tr>
            <tr><td style="padding: 8px; font-weight: bold; background-color: #feb2b2;">Date d'exécution:</td><td style="padding: 8px; background-color: #fff5f5;">{execution_date}</td></tr>
        </table>
        
        <h3 style="color: #9b2c2c;">Traceback d'Erreur :</h3>
        <pre style="background-color: #2d3748; color: #f7fafc; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px;">{formatted_traceback}</pre>
    </div>
    """
    try:
        send_email(to=ALERT_EMAIL, subject=email_subject, html_content=email_body)
    except Exception as e:
        logger.warning(f"Impossible d'envoyer l'e-mail d'échec : {e}")

    # --- 2. [CORRECTIF #2] Alerte Directe Webhook Slack/Teams en cas d'échec ---
    if SLACK_WEBHOOK_URL:
        slack_payload = {
            "text": f"🚨 *[CRITICAL] Échec Pipeline MarketPulse*\n"
                    f"• *Tâche :* `{task_id}`\n"
                    f"• *Run ID :* `{run_id}`\n"
                    f"• *Durée d'exécution avant échec :* `{duration_str}`\n"
                    f"• *Erreur :* ```{str(exception)[:300]}```"
        }
        try:
            requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=10)
        except Exception as e:
            logger.warning(f"Impossible d'envoyer le Webhook Slack d'échec : {e}")



# =============================================================================
# FONCTION DE NOTIFICATION SLACK DIRECTE
# =============================================================================

def send_slack_success_notification(**context):
    """
    Envoie un message formaté sur Slack contenant le bilan du pipeline
    avec la fenêtre d'exécution calculée dynamiquement (~186s).
    """
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="collect_rss", key="run_id") or "N/A"
    article_count = ti.xcom_pull(task_ids="collect_rss", key="article_count") or 0
    
    # [CORRECTIF #1] Récupération du temps réel d'exécution
    _, duration_str = calculate_execution_duration(context)
    
    slack_message = (
        f"🎉 *MarketPulse pipeline terminé avec succès !*\n"
        f"• *Run ID :* `{run_id}`\n"
        f"• *Articles traités :* `{article_count}` articles\n"
        f"• *Durée d'exécution globale (SLA) :* `{duration_str}`\n"
        f"• *Exports Power BI :* `/opt/airflow/exports/*.csv` Ready ✅"
    )
    
    # Tentative via Webhook HTTP direct
    if SLACK_WEBHOOK_URL:
        try:
            payload = {"text": slack_message}
            resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Notification Slack Webhook envoyée avec succès.")
                return
        except Exception as e:
            logger.warning(f"Échec de l'envoi direct Webhook Slack : {e}")

    logger.info(f"[SLACK MESSAGE SIMULATION] : {slack_message}")



# ─── Configuration du DAG ────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "rayan_rezgui",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": notify_on_failure,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=45),
}

dag = DAG(
    dag_id="marketpulse_pipeline",
    description="Platforme intelligente pour l’automatisation de la veille stratégique informationnelle ",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 */2 * * *",   # toutes les 2 heures
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["marketpulse", "NLP", "veille", "production"],
)


# ─── Tâche 1 : Collecte RSS ──────────────────────────────────────────────────

def task_collect(**context):
    """Collecte les articles depuis toutes les sources RSS."""
    from collectors.rss_collector import run_collection
    from airflow.exceptions import AirflowSkipException
    import logging, json

    logger = logging.getLogger("airflow.task") # Initialisation
    run_id = context["run_id"].replace(":", "_").replace("+", "_")
    
    try:
        logger.info(json.dumps({"event": "collection_started", "run_id": run_id}))

        # 2. Exécution protégée : chemin mis à jour vers le conteneur    
        stats = run_collection(
            config_path="/opt/airflow/dags/collectors/sources.yaml",
            run_id=run_id,
        )

        if not stats: 
            raise ValueError("Le collecteur a retourné un objet vide (None).")    

        # Partage du résultat via XCom
        total_articles = stats.get("metadata", {}).get("total") or stats.get("count", 0)
        if total_articles == 0:
            logger.warning("Aucun article détecté via les métadonnées. Vérification du retour du collecteur...")
            logger.info(f"Stats reçues: {stats}")
            raise AirflowSkipException("Pas d'articles à traiter aujourd'hui.")

        # On logue l'erreur pour comprendre pourquoi la donnée manque: 'output_file' est la clé maîtresse
        if "output_file" not in stats:
            raise ValueError(f"Le collecteur n'a pas fourni de chemin 'output_file'.")

        # # Partage du résultat via XCom (Clé unique = 'bronze_file')
        context["ti"].xcom_push(key="run_id", value=run_id)
        context["ti"].xcom_push(key="bronze_file", value=stats["output_file"])
        context["ti"].xcom_push(key="article_count", value=total_articles)
        
        # 4. REMPLACEMENT DU PRINT PAR UN LOG JSON STRUCTURÉ pour monitoring
        logger.info(json.dumps({
            "event": "collection_completed",
            "run_id": run_id,
            "metrics":{"article_count": total_articles},
            "path": stats["output_file"]
        }))

        return stats["output_file"]

    except AirflowSkipException:
        raise # # Propagation du skip sans erreur critique

    except Exception as e:
        # # Gestion d'erreur critique : on loggue et on fait échouer la tâche pour debug
        logger.error(json.dumps({"event": "collection_failed", "run_id": run_id, "error": str(e)}))
        raise # # Le 'raise' seul remonte l'exception originale à Airflow

# ─── Tâche 2 : Validation du volume ─────────────────────────────────────────

def task_validate(**context):
    from data_validator import validate_bronze_data
    import json
    import os
    from datetime import datetime

    logger = get_pipeline_logger("task_validate")
    ti = context['ti']

    # 1. Récupération des données
    bronze_file = ti.xcom_pull(task_ids='collect_rss', key='bronze_file')
    article_count = ti.xcom_pull(task_ids='collect_rss', key='article_count')

    # Fallback pour le mode 'tasks test'
    if not bronze_file:
        logger.warning("XCom 'bronze_file' absent, tentative de détection automatique...")
        base_dir = "/opt/airflow/data/bronze"
        today_path = os.path.join(base_dir, datetime.now().strftime("%Y/%m/%d"))
        if os.path.exists(today_path):
            files = [os.path.join(today_path, f) for f in os.listdir(today_path) if f.endswith('.json')]
            if files:
                bronze_file = max(files, key=os.path.getctime)
                # Lecture manuelle pour obtenir le compte si XCom article_count absent
                with open(bronze_file, 'r') as f:
                    data = json.load(f)
                    article_count = len(data['articles'])
                    # DEBUG : Ajoutez ces deux lignes pour comprendre la structure
                    logger.info(f"DEBUG: Type de data = {type(data)}")
                    if isinstance(data, dict):
                        logger.info(f"DEBUG: Clés de data = {data.keys()}")
                logger.info(f"Fichier détecté automatiquement : {bronze_file} ({article_count} articles)")

    # 2. Vérification du volume (Le remplaçant de ton ancien bloc)
    if not article_count or article_count < 5:
        raise ValueError(f"Volume insuffisant : {article_count} articles (min 5 requis).")

    # 3. Vérification de la structure et du contenu
    try:   

        validate_bronze_data(bronze_file)

        # 4. Succès : Push du chemin pour les tâches suivantes
        ti.xcom_push(key="validated_data_path", value=bronze_file)
        logger.info(json.dumps({"event": "validation_success", "file": bronze_file}))    
        return True

    except Exception as e:
        logger.error(json.dumps({"event": "validation_failed", "error": str(e)}))
        raise
# ─── Tâche 1.5 : Enrichissement Silver (NOUVEAU) ──────────────────────────

def task_enrich_content(**context): # <--- Nouvelle fonction wrapper
    from silver_enricher import enrich_articles
    from logger_utils import get_pipeline_logger    
    import json
    import os
    from glob import glob


    logger = get_pipeline_logger("task_enrich_content")
    ti = context['ti']
    # Récupération du fichier Bronze validé
    bronze_file = ti.xcom_pull(task_ids='validate_data', key="validated_data_path") \
    or ti.xcom_pull(task_ids='collect_rss', key="bronze_file") 
    
    # Secours en cas de test isolé Airflow où l'XCom est absent
    if not bronze_file:
        list_of_files = glob('/opt/airflow/data/bronze/**/*.json', recursive=True)
        if list_of_files:
            bronze_file = max(list_of_files, key=os.path.getctime)
            logger.warning(f"XCom absent, fichier détecté automatiquement : {bronze_file}")

    if not bronze_file or not os.path.exists(bronze_file):
        raise FileNotFoundError(f"Fichier Bronze introuvable pour l'enrichissement : {bronze_file}")


    try:
        # Lecture du fichier Bronze en LECTURE SEULE 
        with open(bronze_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Enrichissement avec log de progression
        initial_count = len(data.get("articles", []))
        data["articles"] = enrich_articles(data["articles"])

        # 4. Construction du chemin de destination dans la couche SILVER
        bronze_path = Path(bronze_file)

        # Maintient la même structure sous-dossiers YYYY/MM/DD si présente
        if "/data/bronze/" in str(bronze_path):
            silver_file_path = Path(str(bronze_path).replace("/data/bronze/", "/data/silver/"))
        else:
            silver_file_path = Path("/opt/airflow/data/silver") / bronze_path.name

        # Création automatique du dossier destination Silver
        silver_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 5. Écriture du fichier enrichi dans la couche SILVER (Immutabilité du Bronze préservée)
        with open(silver_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
        logger.info(json.dumps({
            "event": "enrichment_completed",
            "articles_processed": initial_count,
            "bronze_source": str(bronze_path),
            "silver_output": str(silver_file_path)
        }))

        # 6. Push XCom du nouveau fichier Silver enrichi
        ti.xcom_push(key="enriched_file", value=str(silver_file_path))
        return str(silver_file_path)

    except Exception as e:
        logger.error(json.dumps({"event": "enrichment_failed", "error": str(e)}))
        raise # Indispensable pour marquer la tâche comme 'failed' dans Airflow


# ─── Tâche 3 : Preprocessing NLP ─────────────────────────────────────────────

def task_preprocess(**context):
    """Nettoyage texte + calcul embeddings SBERT."""
    from nlp.preprocessor import process_bronze_file
    from data_validator import validate_bronze_data
    from logger_utils import get_pipeline_logger
    import json
    from pathlib import Path
    import os
    from glob import glob
    from datetime import datetime
 
    logger = get_pipeline_logger("task_preprocess")
    ti = context["ti"]

    # 1. Récupération du fichier Silver issu de task_enrich_content
    input_file = ti.xcom_pull(task_ids="enrich_content") \
                 or ti.xcom_pull(task_ids="enrich_content", key="enriched_file") \
                 or ti.xcom_pull(task_ids="collect_rss", key="bronze_file")

    run_id = ti.xcom_pull(task_ids="collect_rss", key="run_id")

    # Fallback pour tests isolés CLI
    if not input_file:
        logger.warning("XCom 'enriched_file' absent, détection automatique...")
        list_of_files = glob('/opt/airflow/data/silver/**/*.json', recursive=True) or \
                        glob('/opt/airflow/data/bronze/**/*.json', recursive=True)
        if list_of_files:
            input_file = max(list_of_files, key=os.path.getctime)
            logger.info(f"Fichier détecté automatiquement : {input_file}")

    if not run_id:
        run_id = f"test_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    try:
        # 2. Traitement NLP à partir du fichier enrichi
        silver_file = process_bronze_file(Path(input_file), run_id)

        # 3. Log de succès structuré
        logger.info(json.dumps({
            "event": "nlp_preprocessing_completed",
            "run_id": run_id,
            "input_file": str(input_file),
            "output": str(silver_file)
        }))

        ti.xcom_push(key="silver_file", value=str(silver_file))
        ti.xcom_push(key="run_id", value=run_id)
        return str(silver_file)

    except Exception as e:
        logger.error(json.dumps({
            "event": "nlp_preprocessing_failed",
            "run_id": run_id,
            "error": str(e)
        }))
        raise


# ─── Tâche 4 : Clustering ────────────────────────────────────────────────────

def task_cluster(**context):
    """Clustering HDBSCAN ou KMeans + labeling automatique."""
    import sys
    from pathlib import Path
    import glob, os

    clusterer_file = Path("/opt/airflow/dags/nlp/clusterer.py")

    if not clusterer_file.exists():
        raise FileNotFoundError(f"Le fichier {clusterer_file} est introuvable.")
    # Dictionnaire d'environnement pour exécuter le code et récupérer les fonctions
    namespace = {}

    # Exécution dynamique du code source du fichier
    exec(clusterer_file.read_text(encoding="utf-8"), namespace)

    # Récupération de la fonction depuis l'espace de noms exécuté
    run_clustering = namespace.get("run_clustering")
    if not run_clustering:
        raise AttributeError("La fonction 'run_clustering' est introuvable dans clusterer.py")
    # 3. Récupération via XCom (par défaut ou clé spécifique)
    ti = context["ti"]
    silver_file = ti.xcom_pull(task_ids="preprocess_nlp", key="silver_file") or ti.xcom_pull(task_ids="preprocess_nlp")

    # 2. Fallback / Détection automatique si XCom absent (ex: test en CLI)
    if not silver_file:
        # Fallback dynamique sans date figée
        files = glob.glob("/opt/airflow/data/silver/**/*.json", recursive=True)
        if files:
            silver_file = max(files, key=os.path.getctime)

    if not silver_file:
        raise ValueError("Aucun fichier silver trouvé pour le clustering.")

    run_id = context["ti"].xcom_pull(task_ids="collect_rss", key="run_id") or "test_run"
    
    gold_file = run_clustering(Path(silver_file), run_id)

    context["ti"].xcom_push(key="gold_file", value=str(gold_file))
    print(f"✅ Clustering → {gold_file}")
    return str(gold_file)


# ─── Tâche 5 : Synthèse LLM ─────────────────────────────────────────────────

def task_synthesize(**context):
    """Génère les synthèses par cluster via l'API LLM."""
    from nlp.synthesizer import enrich_gold_with_syntheses

    ti = context["ti"]
    gold_file = ( ti.xcom_pull(task_ids="cluster_articles", key="gold_file")
    or ti.xcom_pull(task_ids="cluster_articles")
    )

    if not gold_file:
        files = glob('/opt/airflow/data/gold/**/*.json', recursive=True)
        if files:
            gold_file = max(files, key=os.path.getctime)

    run_id = ti.xcom_pull(task_ids="collect_rss", key="run_id") or "test_run"

    report_file = enrich_gold_with_syntheses(Path(gold_file), run_id)

    ti.xcom_push(key="report_file", value=str(report_file))
    logger.info(f"✅ Synthèses LLM terminées → {report_file}")
    return str(report_file)

# ─── Tâche 6 : Export pour Power BI ─────────────────────────────────────────

def task_export_powerbi(**context):
    """
    Exporte les données en CSV structuré pour Power BI.
    Crée 3 tables: articles, clusters, stats.
    """
    import pandas as pd
    from logger_utils import get_pipeline_logger
    
    logger = get_pipeline_logger("task_export")
    # Récupération du run_id en haut pour être sûr de l'avoir même en cas d'erreur
    run_id = context["ti"].xcom_pull(task_ids="collect_rss", key="run_id")

    try:
    
        report_file = context["ti"].xcom_pull(
        task_ids="synthesize_clusters", key="report_file"
    )

        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)

        export_dir = Path("/opt/airflow/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Table articles
        df_articles = pd.DataFrame(report["articles"])
        df_articles["report_date"] = report["generated_at"][:10]
        df_articles.to_csv(export_dir / "articles_latest.csv",
                       index=False, encoding="utf-8-sig")

        # Table clusters
        clusters_rows = []
        for cid, cdata in report["clusters"].items():
            clusters_rows.append({
                "cluster_id": cid,
                "cluster_label": cdata["label"],
                "article_count": cdata["count"],
                "synthesis": cdata.get("synthesis", ""),
                "report_date": report["generated_at"][:10],
            })
        df_clusters = pd.DataFrame(clusters_rows)
        df_clusters.to_csv(export_dir / "clusters_latest.csv",
                       index=False, encoding="utf-8-sig")

        # Table stats (1 ligne = 1 run)
        stats_row = {
            "run_id": run_id,
            "generated_at": report["generated_at"],
            "total_articles": report["stats"]["total_articles"],
            "n_clusters": report["stats"]["n_clusters"],
            "global_summary": report["global_summary"],
        }
        pd.DataFrame([stats_row]).to_csv(
            export_dir / "stats_history.csv",
            mode="a", header=not (export_dir / "stats_history.csv").exists(),
            index=False, encoding="utf-8-sig",
        )


        # -- Log de métriques pour le jury --
        logger.info(json.dumps({
            "event": "export_completed",
            "run_id": run_id,
            "metrics": {
                "articles_exported": len(df_articles),
                "clusters_exported": len(df_clusters),
                "stats_appended": True
            },
            "output_directory": str(export_dir)
        }))

        return str(export_dir)

    except Exception as e:
        logger.error(json.dumps({
            "event": "export_failed",
            "run_id": run_id,
            "error_message": str(e)
        }))
        raise
# ─── Tâche 7 : Notification ──────────────────────────────────────────────────

def task_notify_email(**context):
    # Envoie un e-mail de type briefing exécutif (C-Level) incluant les KPIs,
    # la synthèse globale (TL;DR) et le Top 5 des clusters thématiques.
    ti = context.get("ti")
    if not ti:
        raise ValueError("Le TaskInstance (ti) est introuvable dans le contexte d'exécution Airflow.")

    run_id = ti.xcom_pull(task_ids="collect_rss", key="run_id") or "N/A"
    article_count = ti.xcom_pull(task_ids="collect_rss", key="article_count") or 0
    # Calcul de la fenêtre d'exécution globale
    _, duration_str = calculate_execution_duration(context)

    # Valeurs par défaut de repli (Fallback)
    global_summary = "Synthese globale non disponible pour ce run."
    total_valid_clusters = 0
    noise_count = 0
    top_clusters_html = ""

    try:
        # Récupération automatique du dernier rapport JSON produit dans le dossier gold
        report_file = ti.xcom_pull(task_ids="synthesize_clusters")
        # Fallback de secours si l'XCom est absent (ex: exécution isolée CLI)
        if not report_file or not os.path.exists(report_file):
            logger.warning(f"⚠️ XCom vide ou fichier introuvable via XCom ({report_file}). Lancement du scan glob de secours...")
            report_files = glob.glob('/opt/airflow/data/gold/reports/*.json')
            if report_files:
                report_file = max(report_files, key=os.path.getctime)

        if report_file and os.path.exists(report_file):
            logger.info(f"📂 Chargement du rapport Gold depuis : {report_file}")
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            global_summary = report_data.get("global_summary", global_summary)
            # Récupération sécurisée du nombre total d'articles depuis les stats du rapport si l'XCom amont est vide
            stats = report_data.get("stats", {})
            if article_count == 0:
                article_count = stats.get("total_articles", 0)

            clusters_raw = report_data.get("clusters", {})
            valid_clusters = {}

            # [CORRECTION HYBRIDE] : Gestion robuste que 'clusters' soit un dict ou une liste
            if isinstance(clusters_raw, dict):
                for cid, cdata in clusters_raw.items():
                    str_cid = str(cid)
                    if str_cid in ("-1", "-1.0"):
                        noise_count = cdata.get("count", 0) if isinstance(cdata, dict) else 0
                    else:
                        valid_clusters[str_cid] = cdata if isinstance(cdata, dict) else {}
            elif isinstance(clusters_raw, list):
                for item in clusters_raw:
                    if not isinstance(item, dict):
                        continue
                    cid = str(item.get("cluster_id", item.get("id", item.get("cluster", "0"))))
                    if cid in ("-1", "-1.0"):
                        noise_count = item.get("count", 0)
                    else:
                        valid_clusters[cid] = item

            total_valid_clusters = len(valid_clusters)

            # Tri et sélection du Top 5 des clusters par volume d'articles décroissant
            sorted_clusters = sorted(
                valid_clusters.items(), 
                key=lambda x: x[1].get("count", 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )[:5]

            for cid, cdata in sorted_clusters:
                label = cdata.get("label", f"Cluster {cid}")
                count = cdata.get("count", 0)
                synthesis = cdata.get("synthesis", "Pas de synthese disponible.")

                top_clusters_html += (
                    '<li style="margin-bottom: 12px;">'
                    f'<strong>[+] {label}</strong> <em>({count} articles)</em><br>'
                    f'<span style="color: #444; font-size: 0.95em;">{synthesis}</span>'
                    '</li>'
                )
        else:
            logger.error(f"❌ Échec critique : Aucun fichier de rapport JSON valide trouvé pour le run {run_id}.")
    except Exception as e:
        logger.error(f"❌ Erreur critique lors du parsing du rapport JSON : {e}")

    # Sujet de l'e-mail orienté C-Level en pur ASCII sécurisé
    subject = f"[MarketPulse] Executive Briefing | {article_count} Articles | {total_valid_clusters} Themes"

    # Corps HTML structuré sécurisé avec parenthèses et concaténation
    body = (
        '<div style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px;">'
        '<h2 style="color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 8px;">[MarketPulse] Executive Briefing</h2>'
        f'<p><strong>Run ID :</strong> {run_id}</p>'
        '<div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 12px; margin: 15px 0; border-radius: 4px;">'
        '<strong>Indicateurs Cles :</strong><br>'
        f'- Articles analyses : <strong>{article_count}</strong><br>'
        f'- Macro-tendances (Clusters) : <strong>{total_valid_clusters}</strong><br>'
        f'- Elements de bruit (Outliers) : <strong>{noise_count}</strong><br>'
        f'- Duree d execution (SLA) : <strong>{duration_str}</strong>'
        '</div>'
        '<h3 style="color: #333; margin-top: 20px;">Synthese Strategique du Jour</h3>'
        f'<p style="background-color: #e9ecef; padding: 12px; border-radius: 6px; font-style: italic; line-height: 1.4;">"{global_summary}"</p>'
        '<h3 style="color: #333; margin-top: 20px;">Top 5 des Thematiques Cles</h3>'
        f'<ul style="padding-left: 20px; line-height: 1.5;">{top_clusters_html if top_clusters_html else "<li>Aucun detail de cluster disponible.</li>"}</ul>'
        '<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 20px 0;">'
        '<p style="font-size: 0.85em; color: #666; text-align: center;">Rapport genere automatiquement par le pipeline MLOps MarketPulse.<br>Systeme souverain de veille strategique - 100% local.</p>'
        '</div>'
    )

    try:
        send_email(to=ALERT_EMAIL, subject=subject, html_content=body)
        logger.info("✅ E-mail exécutif C-Level envoyé avec succès !")
    except Exception as e:
        logger.error(f"❌ Erreur d'envoi de l'e-mail de succès : {e}")
# ─── Définition des tâches ───────────────────────────────────────────────────

with dag:
    # 1. Définition des tâches
    t_collect = PythonOperator(task_id="collect_rss", python_callable=task_collect)
    t_validate = PythonOperator(task_id="validate_data", python_callable=task_validate)
    t_enrich = PythonOperator(task_id="enrich_content", python_callable=task_enrich_content)
    t_preprocess = PythonOperator(task_id="preprocess_nlp", python_callable=task_preprocess)
    t_cluster = PythonOperator(task_id="cluster_articles", python_callable=task_cluster)
    t_synthesize = PythonOperator(task_id="synthesize_clusters", python_callable=task_synthesize)
    t_export = PythonOperator(task_id="export_powerbi", python_callable=task_export_powerbi)

    # Tâche d'alerte Slack/Teams active via SlackWebhookOperator ou Python Callable
    if SLACK_OPERATOR_AVAILABLE and os.getenv("AIRFLOW_CONN_SLACK_DEFAULT"):
        t_notify_slack = SlackWebhookOperator(
            task_id="notify_slack",
            slack_conn_id="slack_default",
            message="🎉 *MarketPulse terminé !* Articles traités et exports mis à jour.",
            trigger_rule="all_success",
        )
    else:
        t_notify_slack = PythonOperator(
            task_id="notify_slack",
            python_callable=send_slack_success_notification,
            trigger_rule="all_success",
        )

    t_notify_email = PythonOperator(
        task_id="notify_email",
        python_callable=task_notify_email,
        trigger_rule="all_success",
    )
    # 2. Flux d'exécution (Le flux d'acier)
    t_collect >> t_validate >> t_enrich >> t_preprocess >> t_cluster >> t_synthesize >> t_export >> t_notify_slack >> t_notify_email
