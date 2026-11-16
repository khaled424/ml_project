# Guide complet de révision MLOps

## Projet de prédiction du churn client — Ateliers pratiques

**Environnement de référence :** WSL Ubuntu, Python 3.12, Docker Desktop, Docker Compose  
**Date de rédaction :** 16 novembre 2026  
**Périmètre :** notebook et pipeline ML, orchestration Prefect, API FastAPI, suivi MLflow, conteneurs Docker, interface Streamlit et monitoring Elasticsearch/Kibana.

> Ce guide décrit le dépôt et les exécutions réellement réalisés. Les pistes Metricbeat, Filebeat, Logstash, Prometheus, Grafana et la création manuelle d'un tableau de bord sont des extensions, pas des composants déjà configurés dans le projet.

## Table des matières

1. [Objectifs et architecture globale](#1-objectifs-et-architecture-globale)
2. [Structure du dépôt et rôle des fichiers](#2-structure-du-dépôt-et-rôle-des-fichiers)
3. [Ateliers de modélisation et de modularisation](#3-ateliers-de-modélisation-et-de-modularisation)
4. [Atelier 3 — Orchestration avec Prefect](#4-atelier-3--orchestration-avec-prefect)
5. [Atelier 4 — API FastAPI et interface Streamlit](#5-atelier-4--api-fastapi-et-interface-streamlit)
6. [Atelier 5 — Suivi et registre des modèles avec MLflow](#6-atelier-5--suivi-et-registre-des-modèles-avec-mlflow)
7. [Atelier 6 — Docker, images et Compose](#7-atelier-6--docker-images-et-compose)
8. [Atelier 7 — Monitoring MLflow, Elasticsearch et Kibana](#8-atelier-7--monitoring-mlflow-elasticsearch-et-kibana)
9. [Démarrer le projet de bout en bout](#9-démarrer-le-projet-de-bout-en-bout)
10. [Résultats et vérifications observés](#10-résultats-et-vérifications-observés)
11. [Révision rapide pour l'examen](#11-révision-rapide-pour-lexamen)
12. [Dépannage et précautions](#12-dépannage-et-précautions)
13. [Documents de référence](#13-documents-de-référence)

## 1. Objectifs et architecture globale

Le projet prédit si un client de banque va quitter la banque. La cible du jeu de données est `Exited` :

- `0` : client resté ;
- `1` : client parti (churn).

Le parcours MLOps va de l'analyse du notebook à une application accessible par HTTP, suivie par MLflow et Kibana :

```text
CSV -> préparation -> entraînement -> évaluation -> modèle joblib
                              |                 |
                              v                 v
                         MLflow Tracking   FastAPI /predict
                                               ^
                                               |
                                      interface Streamlit

MLflow metrics -> Elasticsearch -> Kibana
Prefect orchestre certaines étapes; Docker Compose démarre les services.
```

Le projet illustre plusieurs responsabilités complémentaires :

| Domaine | Question à laquelle il répond | Outil principal |
|---|---|---|
| Machine Learning | Comment préparer les données et mesurer le modèle ? | pandas, scikit-learn |
| Orchestration | Dans quel ordre lancer les tâches et comment suivre leur état ? | Prefect |
| Serving | Comment demander une prédiction à une application ? | FastAPI, Uvicorn |
| Expérimentation | Quels paramètres, métriques et artefacts ont produit ce modèle ? | MLflow |
| Packaging et déploiement | Comment rendre l'environnement portable et démarrer les services ? | Docker, Compose |
| Monitoring | Où rechercher les métriques et comment visualiser leur évolution ? | Elasticsearch, Kibana |

## 2. Structure du dépôt et rôle des fichiers

```text
ml_project/
├── data/
│   ├── Churn_Modelling.csv          # données sources : 10 000 observations
│   └── customer_churn.ipynb         # notebook exploratoire d'origine
├── artifacts/
│   ├── classifier.joblib            # classifieur sérialisé pour l'API
│   └── prepared_data.joblib         # jeux train/test intermédiaires
├── tests/
│   ├── test_model_pipeline.py       # tests unitaires du pipeline ML
│   └── test_environment.py          # test de l'environnement
├── docs/
│   ├── atelier_3_prefect.md
│   ├── atelier_4_fastapi.md
│   ├── atelier_5_mlflow.md
│   ├── atelier_6_docker.md
│   ├── atelier_7_monitoring.md
│   ├── guide_revision_mlops.md      # source de ce PDF
│   └── supports PDF des ateliers
├── requirements.txt                 # manifeste unique de dépendances
├── model_pipeline.py                # fonctions réutilisables du modèle
├── main.py                          # point d'entrée de préparation/train/evaluate
├── pipeline_prefect.py              # flows Prefect (qualité, train, API, CD...)
├── deploiement_prefect.py           # déploiement planifié de flow_all
├── app.py                           # API FastAPI
├── frontend.py                      # interface Streamlit
├── monitoring_pipeline.py           # run MLflow et indexation Elasticsearch
├── start_mlflow_ui.py               # UI MLflow locale sur le port 5000
├── Dockerfile                       # image API multi-stage
├── Dockerfile.frontend              # image Streamlit
├── Dockerfile.mlflow                # image du serveur MLflow
├── compose.yaml                     # cinq services et volume MLflow
├── .dockerignore                    # fichiers exclus du contexte de build
└── .gitignore                       # artefacts locaux exclus de Git
```

Les scripts et fichiers Docker restent à la racine pour conserver des commandes courtes. Le manifeste `requirements.txt` rassemble les dépendances de développement, d'API, de Streamlit, de MLflow et du client Elasticsearch. Les Dockerfiles utilisent ce manifeste comme fichier de **contraintes**, mais installent seulement les dépendances runtime nécessaires à leur image.

Les dossiers `venv/`, `__pycache__/`, `.pytest_cache/`, `.prefect/`, `mlruns/`, `mlartifacts/`, la base locale `mlflow.db` et les artefacts joblib sont générés localement ou ignorés par Git. Le volume Docker `mlflow_data` est distinct des fichiers MLflow locaux.

## 3. Ateliers de modélisation et de modularisation

### 3.1 Du notebook aux fonctions Python

Le notebook sert à explorer le CSV, comprendre ses colonnes et expérimenter. Les cellules qui constituent le pipeline reproductible ont été extraites dans `model_pipeline.py`, et `main.py` fournit le point d'entrée.

| Fonction | Entrée | Sortie / responsabilité |
|---|---|---|
| `load_data(data_path)` | chemin CSV | DataFrame chargé; valide fichier présent et non vide |
| `preprocess_data(df)` | DataFrame brut | `X` numérique et `y=Exited`; encode Gender et écarte les identifiants/champs non utilisés |
| `split_data(X, y, ...)` | features et cible alignées | `X_train`, `X_test`, `y_train`, `y_test` stratifiés |
| `prepare_data(data_path, ...)` | CSV et paramètres de séparation | chaîne `load_data` → `preprocess_data` → `split_data` |
| `train_model(X_train, y_train)` | données d'entraînement | `RandomForestClassifier` ajusté |
| `evaluate_model(model, X_test, y_test)` | modèle et test | accuracy, prédictions, rapport de classification, matrice de confusion |
| `save_model(model, path)` / `load_model(path)` | modèle et chemin | sérialisation/rechargement joblib |
| `train_and_log_model(...)` | train/test et chemin modèle | entraînement, métriques et modèle journalisés dans MLflow, modèle local sauvegardé |

Les fonctions valident les types, les dimensions, les valeurs manquantes et certains paramètres, et documentent leur contrat par des docstrings. Les tests emploient de petits DataFrames afin de vérifier les cas normaux et plusieurs entrées invalides.

### 3.2 Données et choix de prétraitement

Le CSV comporte 10 000 lignes et 14 colonnes. Une observation correspond à un client. Après préparation, les neuf variables explicatives sont :

`CreditScore`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.

`Exited` est la cible. `Gender` est encodée par `LabelEncoder`; `Surname`, `Geography`, `RowNumber` et `CustomerId` ne sont pas transmises au classifieur. Le projet réserve 20 % au test (`test_size=0.2`), utilise une séparation stratifiée, et rend cette séparation reproductible (`random_state=1`). La Random Forest utilise 100 arbres et `random_state=42`.

**Limite importante à expliquer à l'examen :** `Geography` est supprimée et n'a donc pas d'influence sur la prédiction. Le choix du prétraitement et la parité entre les features du modèle et celles envoyées à l'API sont essentiels.

### 3.3 Entraînement, évaluation et persistance

`main.py` exécute tout le pipeline avec `python main.py`, ou séparément :

```bash
python main.py --prepare
python main.py --train
python main.py --evaluate
```

La première étape sauvegarde le tuple des jeux train/test dans `artifacts/prepared_data.joblib`; les deux suivantes le rechargent. L'entraînement sauvegarde le modèle dans `artifacts/classifier.joblib`. `evaluate` a donc besoin des artefacts générés par les étapes précédentes.

L'accuracy mesurée est de **85,25 %**. Pour la classe churn (1), precision ≈ **0,74**, recall ≈ **0,43**, F1 ≈ **0,54**. La matrice de confusion reportée est :

| | Prédit 0 | Prédit 1 |
|---|---:|---:|
| Réel 0 | 1 532 | 61 |
| Réel 1 | 234 | 173 |

L'accuracy seule cache ici 234 départs non détectés. Le recall est crucial si le besoin métier est de détecter un client avant son départ; le seuil de décision, les faux positifs et les coûts métier doivent également être discutés.

## 4. Atelier 3 — Orchestration avec Prefect

**Task** (`@task`) est une unité de travail suivie individuellement (installation, lint, test, préparation, entraînement, API, build/push). **Flow** (`@flow`) décrit l'enchaînement. Prefect enregistre l'état et les logs des runs, permet le suivi dans une UI et prend en charge les déploiements planifiés.

`pipeline_prefect.py` expose ces flows :

| Flow | But |
|---|---|
| `code` | mise à jour configurée, installation des dépendances, Flake8 et pytest |
| `train` | préparation et entraînement |
| `evaluate` | évaluation des artefacts existants |
| `api` | lancement d'Uvicorn au premier plan pour suivre le processus |
| `all` | contrôle, tests, préparation, entraînement et évaluation |
| `cd` | build Docker de l'API, tag et push sur Docker Hub |

Commandes locales :

```bash
python pipeline_prefect.py --flow code
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow evaluate
python pipeline_prefect.py --flow all
python pipeline_prefect.py --flow api
```

L'UI Prefect est accessible sur `http://localhost:4200` après `prefect server start`. `deploiement_prefect.py` appelle `flow_all.serve(...)` avec une planification quotidienne à 02:00 dans le fuseau `Africa/Tunis`. La procédure d'atelier décrit le serveur, le déploiement et le déclenchement dans des terminaux séparés.

**Précaution liée au code réel :** plusieurs flows appellent `update_project()`, qui exécute `git pull --ff-only origin main`. Avant d'exécuter un flow concerné, vérifiez l'état Git et protégez les modifications locales non validées.

## 5. Atelier 4 — API FastAPI et interface Streamlit

### 5.1 API de prédiction

`app.py` charge `artifacts/classifier.joblib` au démarrage. FastAPI expose :

- `GET /` : contrôle de santé, retourne `{"status": "online"}` ;
- `POST /predict` : valide un profil avec Pydantic et renvoie la classe, une interprétation et la probabilité de churn ;
- `/docs` : documentation interactive Swagger/OpenAPI ;
- `/redoc` : documentation ReDoc.

Exemple de corps de requête :

```json
{
  "CreditScore": 650,
  "Gender": "Female",
  "Age": 42,
  "Tenure": 3,
  "Balance": 0.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 100000.0
}
```

Le code convertit `Gender` vers l'encodage attendu par le modèle, aligne les colonnes, puis utilise `predict()` et `predict_proba()`. L'API en conteneur utilise le port hôte **8001**; un lancement local Uvicorn utilise habituellement **8000**.

### 5.2 Interface Streamlit

`frontend.py` présente un formulaire web, construit le JSON et appelle l'API par HTTP. La variable `API_URL` permet de configurer l'adresse. Dans Compose, le frontend utilise `http://api:8000/predict` sur le réseau Docker; depuis WSL, l'URL publiée de l'API est `http://127.0.0.1:8001`.

## 6. Atelier 5 — Suivi et registre des modèles avec MLflow

Les quatre composants à distinguer :

1. **Tracking** : expériences, runs, paramètres, métriques, tags et artefacts.
2. **Projects** : définition reproductible d'un projet et de ses dépendances/points d'entrée.
3. **Models** : format de paquet avec métadonnées `MLmodel` et flavors adaptés aux outils ML.
4. **Model Registry** : versions d'un modèle nommé et cycle de gouvernance (développement, staging, production, archivage).

Le serveur distingue le **backend store**, qui conserve les métadonnées de runs, et l'**artifact store**, qui conserve les fichiers (modèles, rapports, etc.). Le code journalise notamment les paramètres du Random Forest, accuracy/precision/recall/F1, `confusion_matrix.json` et le modèle.

Le nom de l'expérience est `Churn_Prediction_Experiment`; le modèle enregistré est `ChurnRandomForestModel`.

Deux modes existent et ne partagent pas nécessairement leurs données :

| Mode | URI / commande | Persistance |
|---|---|---|
| MLflow local | `python start_mlflow_ui.py`, UI sur `127.0.0.1:5000` | `mlflow.db` et artefacts locaux |
| MLflow sous Compose | `http://127.0.0.1:5001` | volume Docker `mlflow_data` |

Pour envoyer l'entraînement local au serveur Compose :

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python main.py
```

Le script `monitoring_pipeline.py` utilise ce serveur par défaut. Les runs de l'un des serveurs ne doivent pas être supposés visibles dans l'autre.

## 7. Atelier 6 — Docker, images et Compose

**Image** est le modèle immuable construit depuis un Dockerfile; **conteneur** est une instance en cours d'exécution; **registre** (par exemple Docker Hub) distribue les images. `-p hôte:conteneur` publie un port, `-d` démarre en arrière-plan, `docker logs` donne les journaux.

### 7.1 Dockerfiles

- `Dockerfile` : build multi-stage pour FastAPI. L'étape `builder` crée un virtualenv et installe uniquement FastAPI, joblib, numpy, pandas, scikit-learn et Uvicorn sous les contraintes de `requirements.txt`. L'étape `runtime` copie le virtualenv, `app.py`, `model_pipeline.py` et le modèle; elle expose 8000 et lance Uvicorn.
- `Dockerfile.frontend` : installe `requests` et `streamlit`, copie `frontend.py`, expose 8501.
- `Dockerfile.mlflow` : installe MLflow 3.18.0 et démarre un serveur sur 5000 avec backend SQLite et stockage d'artefacts sous `/mlflow`.

`.dockerignore` évite d'envoyer le CSV, les docs, les tests, le venv et les fichiers de suivi au build; le modèle API attendu est réinclus explicitement. Les exclusions réduisent le contexte de build et empêchent d'empaqueter des fichiers inutiles.

### 7.2 Services du `compose.yaml`

| Service | Image / rôle | Port hôte → conteneur | Santé et stockage |
|---|---|---|---|
| `api` | FastAPI | 8001 → 8000 | healthcheck HTTP `/` |
| `frontend` | Streamlit | 8501 → 8501 | dépend de l'API healthy |
| `mlflow` | Tracking server | 5001 → 5000 | volume nommé `mlflow_data` |
| `elasticsearch` | Elasticsearch 7.17.9 | 9200 → 9200 | mode single-node local |
| `kibana` | Kibana 7.17.9 | 5601 → 5601 | se connecte à `elasticsearch:9200` |

Les ports hôte sont liés à `127.0.0.1`, donc destinés à un usage local. Elasticsearch a la sécurité désactivée pour ce TP; ne réutilisez pas cette configuration non sécurisée comme déploiement public.

La taille observée de l'image API optimisée est passée d'environ **1,78 Go** à environ **705 Mo** avec le build multi-stage. La mesure dépend de Docker et de la plateforme.

Le flow Prefect `cd` construit `khaled_4bi6_mlops:latest`, puis tague et pousse l'image vers Docker Hub avec la version déclarée dans `pipeline_prefect.py`. Une publication initiale du tag `1.0` a été confirmée; le code a ensuite été configuré pour le tag `2.0`. Ne supposez pas que ce dernier tag est publié sans vérifier Docker Hub ou relancer le flow.

## 8. Atelier 7 — Monitoring MLflow, Elasticsearch et Kibana

### 8.1 Les trois dimensions du monitoring

- **Dérive des données (data/feature drift)** : la distribution des entrées change par rapport aux données d'entraînement.
- **Dérive conceptuelle (concept drift)** : la relation entre les entrées `X` et la cible `y` change.
- **Santé système/service** : disponibilité, CPU/RAM, latence, débit, erreurs HTTP et journaux.

Le cours présente trois piliers : infrastructure, service API, qualité métier/modèle. Un score MLflow mesuré sur un jeu de test surveille l'expérience d'entraînement; ce n'est pas à lui seul un détecteur de drift en production.

### 8.2 Implémentation réalisée

`monitoring_pipeline.py` :

1. prépare les données et entraîne le même type de modèle en réutilisant `model_pipeline.py` ;
2. journalise un run et les métriques de test dans le serveur MLflow Compose ;
3. sauvegarde le modèle localement et l'enregistre dans le Model Registry ;
4. vérifie la connexion Elasticsearch et crée `mlflow-metrics` avec son mapping si nécessaire ;
5. indexe un document daté, comprenant le run ID, l'expérience, le statut, les scores et les paramètres.

Démarrer les services concernés et lancer le script :

```bash
sudo sysctl -w vm.max_map_count=262144  # si le noyau le demande
docker compose up -d mlflow elasticsearch kibana
docker compose ps

source venv/bin/activate
python -m pip install -r requirements.txt
python monitoring_pipeline.py
```

La commande `sysctl` règle un paramètre du noyau nécessaire à Elasticsearch dans certains environnements WSL. Le script s'exécute depuis WSL, donc il utilise les ports publiés `127.0.0.1:9200` et `127.0.0.1:5001`.

Le script a été exécuté avec succès le 16 novembre 2026. Résultat observé :

- run MLflow : `335944c172ca4c79b414b5e89003aee0` ;
- accuracy de test : **0,8525** ;
- precision churn : **0,7393** ;
- recall churn : **0,4251** ;
- F1 churn : **0,5398** ;
- document Elasticsearch vérifié : `_oGu5aEBWvViVgZ0MvTO`.

Le script crée automatiquement le mapping; ne lancez pas à nouveau un `PUT /mlflow-metrics` qui tenterait de recréer l'index. Pour vérifier dans Kibana Dev Tools :

```http
GET /mlflow-metrics/_mapping
GET /mlflow-metrics/_search
```

Puis créez une Data View nommée `mlflow-metrics` utilisant `@timestamp`, ouvrez **Discover**, réglez la plage temporelle pour inclure le run et sélectionnez les métriques. La réalisation d'un dashboard visuel Kibana reste une étape manuelle.

### 8.3 Limites et piste d'excellence

Le TP journalise les résultats d'une expérience d'entraînement. Il ne collecte pas encore automatiquement les métriques CPU/RAM, les logs de chaque prédiction, la latence de l'API ni les dérives des entrées en continu. Le cours propose Metricbeat, Filebeat et Logstash; Prometheus/Grafana et Evidently sont des outils complémentaires, non installés ici.

## 9. Démarrer le projet de bout en bout

### 9.1 Préparation initiale

Prérequis : dépôt présent dans WSL sous `~/ml_project`, Python 3.12 recommandé, Docker Desktop démarré et intégré à WSL.

```bash
cd ~/ml_project
python3 -m venv venv       # première installation uniquement
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 9.2 Exécuter les tests et les vérifications

```bash
python -m pytest -v
python -m flake8 --max-line-length=120 \
  model_pipeline.py main.py pipeline_prefect.py app.py monitoring_pipeline.py
docker compose config -q
```

### 9.3 Démarrer les cinq services

```bash
docker compose up --build -d
docker compose ps
docker compose logs --tail=50 api frontend mlflow elasticsearch kibana
```

Attendez que chaque service ait fini son démarrage. Ouvrez ensuite :

| Interface | URL |
|---|---|
| API Swagger | `http://localhost:8001/docs` |
| Streamlit | `http://localhost:8501` |
| MLflow | `http://localhost:5001` |
| Elasticsearch | `http://localhost:9200` |
| Kibana | `http://localhost:5601` |

### 9.4 Tester une prédiction

Dans Streamlit, renseignez le formulaire et cliquez **Prédire le risque**. Ou testez l'API depuis WSL :

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":650,"Gender":"Female","Age":42,"Tenure":3,"Balance":0.0,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":100000.0}'
```

Une réponse valide est un JSON comprenant `prediction`, `interpretation` et `probability_churn`. La valeur de prédiction dépend des données de la requête.

### 9.5 Entraîner et suivre le modèle

Pour envoyer un run lancé depuis WSL à MLflow sous Compose :

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python main.py
```

Pour exécuter l'étape de monitoring de l'Atelier 7 :

```bash
python monitoring_pipeline.py
```

### 9.6 Arrêter sans supprimer les données persistantes

```bash
docker compose down
```

Cette commande arrête et supprime les conteneurs, mais conserve normalement le volume nommé. **N'utilisez pas `docker compose down -v` si vous voulez garder les runs et artefacts MLflow du volume.**

## 10. Résultats et vérifications observés

Les contrôles suivants ont été observés dans l'environnement du projet :

- **13 tests pytest réussis**; quatre avertissements de dépendances/MLflow étaient non bloquants ;
- **Flake8 réussi** ;
- `docker compose config -q` réussi ;
- `python main.py --help` et `monitoring_pipeline.py --help` fonctionnels ;
- images API, frontend et MLflow construites ;
- API, Streamlit, MLflow, Elasticsearch et Kibana ont répondu HTTP 200 lors des contrôles ;
- conteneur API déclaré healthy par son healthcheck ;
- run Atelier 7 et document Elasticsearch effectivement créés et relus.

Les mesures et états dépendent de la date, de la machine, des conteneurs démarrés et du modèle entraîné. Rejouer `monitoring_pipeline.py` produit un nouveau run et une nouvelle version enregistrée du modèle.

## 11. Révision rapide pour l'examen

### Définitions à connaître

| Terme | Réponse courte |
|---|---|
| Entraînement / inférence | L'entraînement apprend le modèle depuis les données; l'inférence applique le modèle à de nouvelles entrées. |
| Train / test | Train ajuste le modèle; test estime sa performance sur des observations réservées. |
| Stratification | Préserve approximativement la proportion des classes dans les partitions. |
| Accuracy | Part des prédictions correctes parmi toutes les prédictions. |
| Precision | Parmi les prédictions positives, part réellement positive. |
| Recall | Parmi les vrais positifs, part détectée par le modèle. |
| F1 | Moyenne harmonique de precision et recall. |
| Task / Flow | Unité de travail / orchestration de tâches dans Prefect. |
| Run / Experiment | Exécution enregistrée / regroupement logique de runs dans MLflow. |
| Backend / Artifact store | Métadonnées MLflow / fichiers et modèles associés. |
| Image / Conteneur | Modèle immuable / instance exécutée de l'image. |
| Data drift / Concept drift | Changement de distribution des entrées / changement de relation X-y. |
| Healthcheck | Vérification de disponibilité d'un service, distincte d'un test métier complet. |

### Questions de synthèse avec réponses attendues

**Pourquoi l'accuracy n'est-elle pas suffisante ici ?**  
La classe churn est minoritaire. Avec un recall autour de 0,43, le modèle manque 234 départs sur 407 dans le test. Il faut regarder precision/recall/F1 et les coûts métier.

**Pourquoi garder `model_pipeline.py` et `main.py` séparés ?**  
Le premier contient la logique métier testable/réutilisable; le second gère l'interface CLI, les arguments et l'enchaînement des étapes.

**Que fait MLflow que joblib ne fait pas ?**  
Joblib sérialise le modèle local. MLflow trace contexte, paramètres, métriques, artefacts, runs et versions du modèle.

**Pourquoi FastAPI + Pydantic ?**  
FastAPI expose des routes HTTP et génère OpenAPI/Swagger; Pydantic valide et convertit la requête.

**Pourquoi un build Docker multi-stage ?**  
Il sépare l'installation/construction de l'environnement de l'image runtime, qui ne conserve que le nécessaire.

**Pourquoi Compose ?**  
Il définit des services, réseaux, ports, dépendances et volumes dans une seule configuration reproductible.

**Que persiste le volume `mlflow_data` ?**  
La base MLflow et les artefacts sous `/mlflow` du conteneur; `docker compose down -v` peut le supprimer.

**Est-ce que l'Atelier 7 détecte déjà le drift en temps réel ?**  
Non. Le script exporte des métriques d'une évaluation à Elasticsearch; la collecte continue des entrées et la détection de dérive restent des extensions.

## 12. Dépannage et précautions

| Symptôme | Vérification / action |
|---|---|
| `no such service: elasticsearch` | Vérifiez que le service a été ajouté à `compose.yaml`, puis relancez `docker compose config -q`. |
| Elasticsearch ne démarre pas | Vérifiez `docker compose logs elasticsearch` et `vm.max_map_count`; WSL/Docker Desktop doit avoir assez de mémoire. |
| Kibana n'est pas encore joignable | Attendez son initialisation et suivez `docker compose logs -f kibana`; le conteneur peut démarrer avant que l'UI soit prête. |
| API unhealthy / modèle absent | Vérifiez que `artifacts/classifier.joblib` existe et est copié par le Dockerfile, puis consultez les logs API. |
| Frontend ne contacte pas l'API | Dans Compose l'adresse inter-conteneurs est `http://api:8000/predict`, pas `localhost:8001`. |
| Runs MLflow introuvables | Vérifiez l'URI: local `5000` et conteneur `5001` sont deux stockages distincts. |
| `train` ou `evaluate` échoue | Exécutez `--prepare` d'abord; vérifiez `artifacts/prepared_data.joblib` et `classifier.joblib`. |
| Échec Prefect sans raison apparente | Inspectez l'échec de chaque task; vérifiez aussi l'effet du `git pull` de `update_project()`. |
| Push Docker Hub refusé | Utilisez `docker login`/un PAT dans le gestionnaire d'identifiants; ne mettez jamais un secret dans le code ou le rapport. |

**Précautions :**

- Ne chargez que des fichiers joblib de confiance : la désérialisation pickle peut exécuter du code.
- Ne publiez pas une API, Kibana ou Elasticsearch de TP tels quels sur Internet. La configuration Elasticsearch a la sécurité désactivée et les services sont destinés à un usage local.
- Ne placez pas de mot de passe ou de Personal Access Token dans Git, Dockerfile, notebook, image ou document.
- Un modèle a besoin du même ordre et encodage de features à l'entraînement et à l'inférence.
- Les performances sur le jeu fourni ne garantissent pas la performance future; surveillez les données, le modèle et le service.

## 13. Documents de référence

- Pipeline et commandes du projet : [`README.md`](../README.md)
- Orchestration : [`atelier_3_prefect.md`](atelier_3_prefect.md)
- API : [`atelier_4_fastapi.md`](atelier_4_fastapi.md)
- MLflow : [`atelier_5_mlflow.md`](atelier_5_mlflow.md)
- Docker : [`atelier_6_docker.md`](atelier_6_docker.md)
- Monitoring : [`atelier_7_monitoring.md`](atelier_7_monitoring.md)
- Notebook : [`data/customer_churn.ipynb`](../data/customer_churn.ipynb)
