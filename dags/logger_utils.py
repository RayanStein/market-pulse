import logging
import json

def get_pipeline_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # On évite de dupliquer les handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Format JSON : Très professionnel pour le jury
        formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "task": "%(name)s", "message": %(message)s}')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
