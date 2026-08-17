# 🔍 MarketPulse

> Pipeline de veille IA — collecte automatique d'articles, clustering NLP,
> synthèses générées par LLM et dashboard Power BI mis à jour toutes les 2h.

---

## Architecture globale

```
Sources RSS/Web
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  Apache Airflow                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Collecte │→│   NLP    │→│    Clustering     │  │
│  │  (Bronze) │  │ (Silver) │  │     (Gold)        │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                        │             │
│                               ┌────────────────┐    │
│                               │  Synthèse LLM  │    │
│                               └────────────────┘    │
└─────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼────────────────┐
                    ▼                   ▼                ▼
              FastAPI REST          MLflow           Power BI
              (port 8000)        (port 5000)        Dashboard
```

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Orchestration | Apache Airflow 2.9 |
| Collecte | feedparser, BeautifulSoup |
| NLP | sentence-transformers (SBERT multilingue) |
| Clustering | HDBSCAN / KMeans |
| LLM | Claude API (Anthropic) |
| Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Visualisation | Power BI |
| Infra | Oracle Cloud Free Tier (A1.Flex) |
| Conteneurs | Docker Compose |

## Installation rapide

```bash
# 1. Cloner le projet
git clone https://github.com/rayan-rezgui/marketpulse.git
cd marketpulse

# 2. Configurer l'environnement
cp .env.example .env
# Éditer .env avec ta clé Anthropic

# 3. Lancer tous les services
docker-compose up -d

# 4. Accéder aux interfaces
# Airflow  : http://localhost:8080  (admin / MarketPulse2024!)
# MLflow   : http://localhost:5000
# FastAPI  : http://localhost:8000/docs
```

## Lancer manuellement le pipeline

```bash
# Via Airflow UI : Trigger DAG "marketpulse_pipeline"
# Ou via CLI :
docker exec marketpulse_airflow airflow dags trigger marketpulse_pipeline
```

## Structure du projet

```
marketpulse/
├── dags/
│   └── marketpulse_dag.py      # DAG Airflow principal
├── collectors/
│   ├── rss_collector.py        # Collecte RSS (Bronze)
│   └── sources.yaml            # Configuration sources
├── nlp/
│   ├── preprocessor.py         # Nettoyage + embeddings (Silver)
│   ├── clusterer.py            # HDBSCAN + labeling (Gold)
│   └── synthesizer.py          # Synthèses LLM
├── api/
│   └── main.py                 # FastAPI endpoints
├── dashboard/
│   └── POWERBI_GUIDE.md        # Guide Power BI + DAX
├── data/
│   ├── bronze/                 # Articles bruts
│   ├── silver/                 # Articles enrichis + embeddings
│   └── gold/                   # Clusters + rapports
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Auteur

**Rayan Rezgui** — Master Data Science, Leaders University Nabeul (2026)
PFE — MSH AND MORE Werbeagentur GmbH, Köln
