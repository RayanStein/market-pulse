"""
MarketPulse — API REST FastAPI
Expose les rapports, articles et clusters pour Power BI et frontend.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


GOLD_PATH = Path(os.getenv("GOLD_PATH", "./data/gold"))

app = FastAPI(
    title="MarketPulse API",
    description="API de veille IA — actualités clustering et synthèses",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_latest_report() -> Optional[dict]:
    """Charge le rapport le plus récent."""
    reports_dir = GOLD_PATH / "reports"
    if not reports_dir.exists():
        return None
    report_files = sorted(reports_dir.glob("report_*.json"), reverse=True)
    if not report_files:
        return None
    with open(report_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "MarketPulse API",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    report = get_latest_report()
    return {
        "status": "ok",
        "last_report": report["generated_at"] if report else None,
        "total_articles": report["stats"]["total_articles"] if report else 0,
    }


@app.get("/report/latest", tags=["Rapport"])
def get_latest():
    """Retourne le dernier rapport complet."""
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="Aucun rapport disponible")
    return report


@app.get("/report/summary", tags=["Rapport"])
def get_summary():
    """Retourne uniquement le résumé global du dernier rapport."""
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="Aucun rapport disponible")
    return {
        "generated_at": report["generated_at"],
        "global_summary": report["global_summary"],
        "stats": report["stats"],
        "next_update": report.get("next_update"),
    }


@app.get("/clusters", tags=["Clusters"])
def get_clusters():
    """Retourne tous les clusters avec leurs synthèses."""
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="Aucun rapport disponible")
    return {
        "generated_at": report["generated_at"],
        "clusters": report["clusters"],
    }


@app.get("/clusters/{cluster_id}", tags=["Clusters"])
def get_cluster(cluster_id: str):
    """Retourne les articles d'un cluster spécifique."""
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="Aucun rapport disponible")

    cluster = report["clusters"].get(cluster_id)
    if not cluster:
        raise HTTPException(
            status_code=404, detail=f"Cluster {cluster_id} introuvable"
        )

    articles = [
        a for a in report["articles"]
        if str(a["cluster_id"]) == cluster_id
    ]
    return {**cluster, "articles": articles}


@app.get("/articles", tags=["Articles"])
def get_articles(
    source: Optional[str] = Query(None, description="Filtrer par source"),
    language: Optional[str] = Query(None, description="Filtrer par langue"),
    cluster_id: Optional[str] = Query(None, description="Filtrer par cluster"),
    limit: int = Query(50, le=200),
):
    """Retourne les articles avec filtres optionnels."""
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="Aucun rapport disponible")

    articles = report["articles"]

    if source:
        articles = [a for a in articles
                    if source.lower() in a["source_name"].lower()]
    if language:
        articles = [a for a in articles if a["language"] == language]
    if cluster_id:
        articles = [a for a in articles
                    if str(a["cluster_id"]) == cluster_id]

    return {
        "total": len(articles),
        "articles": articles[:limit],
    }


@app.get("/powerbi/articles", tags=["Power BI"])
def powerbi_articles():
    """
    Endpoint optimisé pour Power BI (format aplati).
    Compatible DirectQuery via Power BI connector.
    """
    report = get_latest_report()
    if not report:
        return {"value": []}

    rows = []
    for art in report["articles"]:
        rows.append({
            "id": art["id"],
            "title": art["title"],
            "url": art["url"],
            "source": art["source_name"],
            "published_at": art["published_at"],
            "cluster_id": art["cluster_id"],
            "cluster_label": art["cluster_label"],
            "language": art["language"],
            "category": art["category"],
            "keywords": ", ".join(art.get("keywords", [])),
            "x_viz": art.get("x_viz", 0),
            "y_viz": art.get("y_viz", 0),
            "report_date": report["generated_at"][:10],
        })
    return {"value": rows}


@app.get("/powerbi/clusters", tags=["Power BI"])
def powerbi_clusters():
    """Endpoint clusters optimisé pour Power BI."""
    report = get_latest_report()
    if not report:
        return {"value": []}

    rows = []
    for cid, cdata in report["clusters"].items():
        rows.append({
            "cluster_id": cid,
            "label": cdata["label"],
            "article_count": cdata["count"],
            "synthesis": cdata.get("synthesis", ""),
            "sources": ", ".join(cdata.get("top_sources", [])),
            "report_date": report["generated_at"][:10],
        })
    return {"value": rows}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
