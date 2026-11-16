# Atelier 3 : Orchestration de Workflows Machine Learning avec Prefect (Cours & TP)

Ce document rassemble l'intégralité du cours théorique et le guide pratique pas-à-pas pour réussir le **TP3 / Atelier 3 : Orchestration avec Prefect**.

---

## PARTIE 1 : LE COURS - Orchestration et Concepts Prefect

### 1. Pourquoi l'orchestration en MLOps ?
Dans un projet de Machine Learning en production, exécuter des scripts de manière manuelle ou séquentielle devient rapidement instable et non reproductible. 

L'**orchestration** permet de :
- **Automatiser l'enchaînement** de toutes les étapes du cycle de vie ML (vérification du code, préparation des données, entraînement, évaluation).
- **Superviser l'exécution** et suivre l'état de chaque tâche en temps réel.
- **Réagir automatiquement** en cas d'échec (système de tentatives/retries, notifications, logs centralisés).
- **Planifier les exécutions** à des fréquences régulières (ex. réentraînement quotidien du modèle).

---

### 2. Comparatif des Outils de Pipeline CI/CD et d'Orchestration
Alors que des outils généraux de CI/CD comme **Jenkins**, **GitLab CI**, ou **GitHub Actions** sont parfaitement adaptés pour construire et tester le code logiciel, **Prefect** est un orchestrateur spécialement conçu pour les pipelines de données et le Machine Learning sous Python.

#### Avantages principaux de Prefect :
1. **Adoption progressive :** Il suffit d'ajouter de simples décorateurs (`@task`, `@flow`) sur des fonctions Python existantes.
2. **Exécution locale simplifiée :** Les pipelines (flows) peuvent être exécutés et testés directement en local comme n'importe quel script Python.
3. **Clarté et Observabilité :** Une interface Web (UI) native permet d'inspecter visuellement la dépendance entre les tâches, les logs et les résultats.

---

### 3. Les Concepts Clés de Prefect

| Concept | Définition & Rôle |
| :--- | :--- |
| **Task (`@task`)** | L'unité élémentaire de travail (ex: prétraiter les données, entraîner le modèle). Elle gère les retries, la mise en cache et les logs d'exécution. |
| **Flow (`@flow`)** | La fonction Python principale qui orchestre le flux d'exécution et les dépendances entre les différentes *tasks*. |
| **Runs & States** | Chaque exécution (*flow run* ou *task run*) possède un état explicite : `Pending`, `Running`, `Completed`, `Failed`, `Cancelled`. |
| **Deployment** | La configuration qui rend un *flow* exécutable à distance, manuellement ou de manière programmée via un *Scheduler* (ex. planification Cron). |

---

## PARTIE 2 : LE TP - Guide Pratique Pas-à-Pas

### Prérequis
- Avoir validé l'Atelier 2 avec un code modulaire fonctionnel dans `model_pipeline.py` et un point d'entrée dans `main.py`.
- Environnement virtuel Python activé sous WSL / Ubuntu (`source venv/bin/activate`).

---

### Étape 1 : Préparer l'Environnement

1. Activez votre environnement virtuel sous WSL/Ubuntu :
   ```bash
   cd ~/ml_project
   source venv/bin/activate
   ```
2. Installez Prefect :
   ```bash
   pip install prefect
   ```
3. Vérifiez que `prefect` est présent dans `requirements.txt`, puis installez les dépendances avec ce fichier.

---

### Étape 2 : Créer le Fichier de Pipeline (`pipeline_prefect.py`)

Créez le fichier `pipeline_prefect.py` à la racine de votre projet. Ce fichier va encapsuler vos appels de `main.py` et `model_pipeline.py` sous forme de **Tasks** et de **Flows**.

```python
import sys
import subprocess
from pathlib import Path
from prefect import task, flow, get_run_logger

# Chemins de configuration
PYTHON_BIN = sys.executable
MAIN_PY = Path("main.py")

# ==========================================
# 1. TASKS : Définition des tâches unitaires
# ==========================================

@task(name="install-dependencies")
def install_dependencies():
    logger = get_run_logger()
    logger.info("Vérification et installation des dépendances...")
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

@task(name="check-code-quality")
def check_code_quality():
    logger = get_run_logger()
    logger.info("Vérification de la qualité du code (Flake8 / Linter)...")
    # On cible uniquement nos fichiers Python du projet pour optimiser le temps d'exécution
    subprocess.run([PYTHON_BIN, "-m", "flake8", "model_pipeline.py", "main.py", "pipeline_prefect.py"], check=False)

@task(name="run-unit-tests")
def run_unit_tests():
    logger = get_run_logger()
    logger.info("Exécution des tests unitaires avec pytest...")
    subprocess.run([PYTHON_BIN, "-m", "pytest", "-v"], check=True)

@task(name="prepare-data")
def prepare_data_task():
    logger = get_run_logger()
    logger.info("Exécution de la préparation des données...")
    subprocess.run([PYTHON_BIN, str(MAIN_PY), "--prepare"], check=True)

@task(name="train-model")
def train_model_task():
    logger = get_run_logger()
    logger.info("Entraînement et sauvegarde du modèle...")
    subprocess.run([PYTHON_BIN, str(MAIN_PY), "--train"], check=True)

@task(name="evaluate-model")
def evaluate_model_task():
    logger = get_run_logger()
    logger.info("Évaluation des performances du modèle...")
    subprocess.run([PYTHON_BIN, str(MAIN_PY), "--evaluate"], check=True)

# ==========================================
# 2. FLOWS : Structuration des workflows
# ==========================================

@flow(name="code", log_prints=True)
def flow_code():
    """Flow dédié aux contrôles de qualité et de sécurité du code."""
    install_dependencies()
    check_code_quality()
    run_unit_tests()

@flow(name="train", log_prints=True)
def flow_train():
    """Flow dédié à la préparation des données et à l'entraînement."""
    prepare_data_task()
    train_model_task()

@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """Flow dédié à l'évaluation du modèle."""
    evaluate_model_task()

@flow(name="all", log_prints=True)
def flow_all():
    """Flow complet enchaînant toutes les étapes du pipeline MLOps."""
    flow_code()
    prepare_data_task()
    train_model_task()
    evaluate_model_task()

# ==========================================
# 3. CLI ENTRYPOINT : Lancement manuel local
# ==========================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline MLOps Prefect")
    parser.add_argument("--flow", type=str, default="all", choices=["code", "train", "evaluate", "all"])
    args = parser.parse_args()

    flows = {
        "code": flow_code,
        "train": flow_train,
        "evaluate": flow_evaluate,
        "all": flow_all
    }

    flows[args.flow]()
```

---

### Étape 3 : Tester les Flows en Ligne de Commande (Local CLI)

Avant de lancer le serveur de déploiement, validez que vos flows s'exécutent sans erreur depuis votre terminal :

```bash
# Exécution du flow complet
python pipeline_prefect.py --flow all

# Exécution ciblée du flow d'entraînement
python pipeline_prefect.py --flow train

# Exécution ciblée du flow d'évaluation
python pipeline_prefect.py --flow evaluate
```

---

### Étape 4 : Automatisation et Déploiement (Architecture 3 Terminaux)

Pour simuler un environnement de production complet avec interface de supervision (UI), serveur API et exécution planifiée, ouvrez **3 terminaux WSL distincts**.

#### Pourquoi 3 terminaux ?
- **Terminal 1 (Le Serveur) :** Fait tourner la base de données de suivi Prefect et l'interface Web UI (`http://localhost:4200`).
- **Terminal 2 (Le Worker / Agent) :** Écoute les requêtes de déploiement et exécute physiquement le code sur votre machine.
- **Terminal 3 (Le Déclencheur CLI) :** Déclenche les exécutions manuellement à la demande ou vérifie la liste des *deployments*.

---

#### 1. Créer le script de déploiement (`deploiement_prefect.py`)

Créez le fichier `deploiement_prefect.py` pour enregistrer vos workflows auprès du serveur Prefect et définir une planification automatisée (*Cron*) :

```python
from prefect.client.schemas.schedules import CronSchedule
from pipeline_prefect import flow_all, flow_train, flow_evaluate

if __name__ == "__main__":
    # Déploiement du pipeline complet planifié tous les jours à 02:00 AM (Heure de Tunis)
    flow_all.serve(
        name="ml-pipeline-all",
        schedules=[
            CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis")
        ],
        tags=["mlops", "full-pipeline"]
    )
```

---

#### 2. Procédure d'exécution dans les 3 Terminaux

##### **Terminal 1 : Lancement du Serveur et de l'UI**
```bash
cd ~/ml_project
source venv/bin/activate
prefect server start
```
> 🌐 Ouvrez votre navigateur et accédez à **http://localhost:4200** pour visualiser le tableau de bord Prefect.

##### **Terminal 2 : Enregistrement et Lancement du Worker**
```bash
cd ~/ml_project
source venv/bin/activate

# Pointer l'agent vers l'API du serveur local
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

# Lancer le script de déploiement
python deploiement_prefect.py
```

##### **Terminal 3 : Déclenchement Manuel & Contrôle CLI**
```bash
cd ~/ml_project
source venv/bin/activate

# Lister les déploiements actifs
prefect deployment ls

# Déclencher manuellement l'exécution du pipeline complet
prefect deployment run 'all/ml-pipeline-all'
```

---

### Étape 5 : Récapitulatif des Livrables de l'Atelier 3

1. **`pipeline_prefect.py`** : Code définissant l'ensemble des tâches (`@task`) et des flux (`@flow`).
2. **`deploiement_prefect.py`** : Script d'enregistrement de déploiement avec planification Cron.
3. **Interface Web UI** : Supervision en temps réel des exécutions et états (*Completed*, *Failed*) sur `http://localhost:4200`.
