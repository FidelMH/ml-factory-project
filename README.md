# ML Factory — Zero-Downtime Model Serving

Infrastructure MLOps permettant de mettre à jour un modèle de Machine Learning en production **sans redémarrer aucun conteneur**.

## Architecture

```
ml-factory-project/
├── src/
│   ├── api/        # FastAPI — serving des prédictions
│   ├── front/      # Streamlit — interface utilisateur
│   └── train/      # Script d'entraînement et d'enregistrement
├── data/           # iris_test.csv
├── docker-compose.yml
└── .env
```

| Service | Rôle | Port |
|---|---|---|
| **MLflow** | Model Registry + Tracking | `5000` |
| **MinIO** | Object Storage (artefacts S3) | `9000` / `9001` |
| **API** | FastAPI — prédictions | `8000` |
| **Front** | Streamlit — UI | `8501` |

## Prérequis

- Docker + Docker Compose
- Python 3.11+ avec [uv](https://docs.astral.sh/uv/)

## Démarrage

### 1. Variables d'environnement

Copier le fichier d'exemple et renseigner les valeurs :

```bash
cp .env.example .env
```

### 2. Lancer l'infrastructure

```bash
docker compose up -d
```

### 3. Entraîner et enregistrer un modèle

```bash
uv run src/train/train.py
```

Le script entraîne un modèle, le pousse dans MinIO, l'enregistre dans le Model Registry MLflow et lui attribue automatiquement l'alias `Production`.

### 4. Accéder aux services

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| API (docs) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| MinIO Console | http://localhost:9001 |

## Fonctionnement : Zero-Downtime

L'API vérifie à chaque requête si l'alias `production` pointe vers une nouvelle version. Si oui, elle recharge le modèle à chaud depuis MinIO — sans redémarrage.

### Scénario de validation

**Phase 1 — Automatique** : entraîner une Régression Logistique. L'alias `Production` est assigné automatiquement. Streamlit affiche Version 1.

**Phase 2 — Manuel** : entraîner un RandomForest sans auto-alias. Aller sur l'UI MLflow et assigner manuellement l'alias `Production` à la Version 2. Streamlit bascule instantanément.

## API

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Health check |
| `/model_infos` | GET | Version du modèle en production |
| `/predict` | POST | Prédiction Iris |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

```json
{
  "prediction": 0,
  "probabilities": [0.97, 0.02, 0.01],
  "version": "1"
}
```
