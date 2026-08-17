#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Lancement de l'infrastructure Market Pulse..."
echo "=================================================="

# 1. Nettoyage propre au cas où des vieux conteneurs trainent
docker compose down

# 2. Lancement en premier de la base de données, mlflow, ollama et l'api
echo "📦 Démarrage des services de base (Postgres, MLflow, Ollama, API)..."
docker compose up -d postgres mlflow ollama api

# 3. Attente active que Postgres soit réellement prêt et healthy
echo "⏳ Attente de la santé de la base de données PostgreSQL..."
until docker inspect --format='{{json .State.Health.Status}}' marketpulse_postgres_v2 | grep -q "healthy"; do
  sleep 2
  echo -n "."
done
echo -n -e "\n"
echo "✅ PostgreSQL est opérationnel !"

# 4. Lancement d'Airflow (webserver et scheduler)
echo "🌪️ Démarrage des services Apache Airflow..."
docker compose up -d airflow-webserver airflow-scheduler

echo "=================================================="
echo "🎉 Tout est prêt ! L'application est disponible."
echo "=================================================="
