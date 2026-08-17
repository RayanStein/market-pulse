# MarketPulse — Guide Power BI Dashboard

## 1. Connexion Power BI → API MarketPulse

### Via Web Connector (recommandé)
1. Power BI Desktop → Obtenir des données → Web
2. URL : `http://<IP_ORACLE>:8000/powerbi/articles`
3. Format : JSON → Convertir en table
4. Répéter pour : `http://<IP_ORACLE>:8000/powerbi/clusters`

### Rafraîchissement automatique
- Accueil → Transformer les données → Propriétés de requête
- Cocher "Inclure dans l'actualisation planifiée"
- Power BI Service → Planifier l'actualisation : toutes les 2 heures

---

## 2. Modèle de données (Relations)

```
articles [cluster_id] ──── clusters [cluster_id]
    │
    └── Cardinalité : N:1
        Direction filtre : Articles → Clusters
```

---

## 3. Mesures DAX essentielles

### Mesures de base
```dax
-- Nombre total d'articles
Total Articles = COUNTROWS(articles)

-- Articles aujourd'hui
Articles Today =
CALCULATE(
    COUNTROWS(articles),
    articles[report_date] = TODAY()
)

-- Nombre de clusters actifs
Clusters Actifs = DISTINCTCOUNT(articles[cluster_id])

-- Sources uniques
Sources Uniques = DISTINCTCOUNT(articles[source])
```

### Mesures avancées
```dax
-- Articles par cluster (pour jauge)
Articles Par Cluster =
DIVIDE([Total Articles], [Clusters Actifs], 0)

-- % d'articles par langue
Pct Français =
DIVIDE(
    CALCULATE(COUNTROWS(articles), articles[language] = "fr"),
    [Total Articles],
    0
)

-- Top cluster (label du plus grand)
Top Cluster Label =
CALCULATE(
    SELECTEDVALUE(clusters[label]),
    TOPN(1, clusters, clusters[article_count], DESC)
)

-- Évolution articles (vs run précédent)
Articles Growth =
VAR current = [Total Articles]
VAR previous =
    CALCULATE(
        COUNTROWS(articles),
        DATEADD(articles[report_date], -1, DAY)
    )
RETURN
IF(previous = 0, BLANK(), DIVIDE(current - previous, previous))
```

---

## 4. Visuels recommandés

### Page 1 — Vue d'ensemble
| Visuel | Données | Notes |
|--------|---------|-------|
| Carte KPI | Total Articles | Grand format, couleur accent |
| Carte KPI | Clusters Actifs | Couleur secondaire |
| Carte KPI | Sources Uniques | Couleur tertiaire |
| Graphique barres | Articles par cluster_label | Trié décroissant |
| Graphique secteurs | Articles par language | Max 5 langues |
| Texte dynamique | global_summary (API) | Zone de texte enrichi |

### Page 2 — Scatter Plot Thématique
| Visuel | Config |
|--------|--------|
| Nuage de points | X = x_viz, Y = y_viz |
| Taille des bulles | Nombre d'articles |
| Couleur | cluster_label |
| Info-bulles | title, source, published_at |

> Ce scatter plot montre visuellement les clusters sémantiques.

### Page 3 — Détail des clusters
| Visuel | Config |
|--------|--------|
| Slicer | cluster_label |
| Table | title, source, published_at, url |
| Carte texte | synthesis (résumé LLM) |
| Graphique courbe | Articles par heure de publication |

### Page 4 — Historique & tendances
| Visuel | Config |
|--------|--------|
| Courbe | Total Articles dans le temps (report_date) |
| Aires empilées | Articles par catégorie dans le temps |
| Tableau | Historique des runs (run_id, n_clusters, total) |

---

## 5. Mise en forme recommandée

```
Thème couleurs :
  Primaire  : #185FA5  (bleu)
  Secondaire: #1D9E75  (vert)
  Tertiaire : #7F77DD  (violet)
  Neutre    : #F5F7FA  (fond)
  Texte     : #1A1A2E

Police : Segoe UI (Power BI default)
Fond canvas : #F5F7FA
Arrière-plan visuels : Blanc avec ombre légère
```

---

## 6. Publication & partage

1. Power BI Service → Publier le rapport
2. Créer un **workspace** "MarketPulse"
3. Configurer l'actualisation : toutes les 2h (aligner avec Airflow)
4. Créer un **Dashboard** épinglant les 4 KPIs principaux
5. Partager via lien ou intégrer dans une page web (`Fichier → Incorporer`)
