import logging
import json
from pathlib import Path

# Initialisation du logger Airflow
logger = logging.getLogger("airflow.task")

def validate_bronze_data(file_path):
    """
    Valide le fichier JSON situé au chemin 'file_path'.
    Ce chemin est transmis dynamiquement par le DAG via XCom.
    """
    path = Path(file_path)
    
    # 1. Vérification de l'existence
    if not path.exists():
        error_msg = f"Le fichier attendu est introuvable au chemin : {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"DEBUG: Début de la validation pour le fichier : {path.name}")
    
    # 2. Lecture et parsing du JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de décodage JSON dans le fichier {path.name} : {str(e)}")

    # 3. Validation de la structure (Cohérence avec rss_collector)
    # On s'attend à ce que le collecteur génère une liste sous la clé "articles"
    articles = data.get("articles", [])
    
    if not isinstance(articles, list):
        raise ValueError(f"Format invalide dans {path.name} : la clé 'articles' doit contenir une liste.")

    count = len(articles)
    logger.info(f"DEBUG: Le fichier {path.name} contient {count} articles.")

    # 4. Validation du volume métier
    if count < 5:
        raise ValueError(f"Volume insuffisant dans {path.name} : seulement {count} articles trouvés (min 5 requis).")

    logger.info(f"SUCCESS: Validation terminée avec succès pour {path.name}.")
    return True
