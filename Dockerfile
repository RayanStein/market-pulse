FROM apache/airflow:2.10.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Créer le dossier de cache et donner la propriété à l'utilisateur airflow
RUN mkdir -p /opt/airflow/models && chown -R airflow:root /opt/airflow/models

USER airflow

# Dossier de cache HF
ENV HF_HOME=/opt/airflow/models
ENV TRANSFORMERS_CACHE=/opt/airflow/models

# Copie et installation en mode global forcé (--system)
COPY --chown=airflow:root requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip "setuptools<74.0.0" wheel
# On force un numpy compatible avec pandas pour éviter l'erreur de binary incompatibility
RUN pip install --no-cache-dir "numpy<2.0.0" "pandas>=2.0.0,<2.3.0"
RUN pip install --no-cache-dir --default-timeout=1000 --no-user -r /requirements.txt
