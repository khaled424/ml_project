# Séance 05 & Atelier 5 : Versioning des Modèles et Gestion du Cycle de Vie ML avec MLflow

---

## 1. Le Cours : Versioning et Cycle de Vie ML avec MLflow

### **A. La Problématique du MLOps sans Versioning**
En développement logiciel classique, Git permet de versionner l'intégralité du code source [11]. Cependant, en **Machine Learning**, le code seul ne suffit pas à définir le comportement de l'application [8]. Le comportement dépend de la combinaison complexe entre :
* **Le Code** (scripts d'entraînement et de prétraitement) [12].
* **Les Données** (jeu d'entraînement, de validation et de test) [8].
* **Les Hyperparamètres** (nombre d'arbres, taux d'apprentissage, profondeur maximale...) [12].
* **L'Environnement matériel et logiciel** (versions des librairies, GPU/CPU) [10].

Sans outil dédié au suivi et au versioning des expériences ML, les équipes sont confrontées à plusieurs problèmes majeurs [8] :
1. **Absence de traçabilité :** Impossibilité de retrouver quels hyperparamètres ou quelles données ont produit un modèle donné en production [8].
2. **Non-reproductibilité :** Difficulté de réexécuter une expérience passée et d'obtenir des résultats identiques [8].
3. **Absence de centralisation :** Multiplication des fichiers de modèle éparpillés sans métadonnées associées [8].
4. **Déploiement complexe :** Absence de format standardisé pour passer d'un modèle exploratoire à un service de production [8].

---

### **B. Qu'est-ce que MLflow ?**
**MLflow** est une plateforme open source développée par **Databricks** (en 2018) conçue pour gérer l'ensemble du cycle de vie des modèles de Machine Learning [8].

**Caractéristiques principales :**
* **Open Source et Agnostique :** Compatible avec tous les frameworks ML (`scikit-learn`, `TensorFlow`, `PyTorch`, `XGBoost`, `ONNX`, etc.) et tous les langages (`Python`, `R`, `Java`, `REST API`) [9, 10].
* **Portabilité :** Fonctionne de manière identique sur une machine locale, un serveur virtuel ou un cluster Cloud [9].
* **Intégration facile :** S'intègre sans modifier l'architecture existante de vos scripts [9].

---

### **C. L'Architecture et les 4 Composants Clés de MLflow**

MLflow est structuré autour de **4 composants fondamentaux** qui répondent chacun à une étape précise du cycle de vie ML [10, 11] :

1. **MLflow Tracking :**
   Composant utilisé lors de l'entraînement pour enregistrer et comparer les exécutions (*runs*) au sein d'expériences (*experiments*) [12] :
   * `log_param(key, value)` : Enregistre les hyperparamètres (ex. `n_estimators=100`, `max_depth=10`) [13].
   * `log_metric(key, value)` : Enregistre les métriques de performance (ex. `accuracy`, `precision`, `recall`, `f1_score`) [13].
   * `log_artifact(path)` : Sauvegarde des fichiers produits (graphiques, rapports) [13, 32].
   * `log_model(model, artifact_path)` : Sauvegarde et sérialise le modèle dans un format standard MLflow [13, 19].
   * `autolog()` : Capture automatique des paramètres et métriques (`mlflow.autolog()`) [13].

2. **MLflow Projects :**
   Format de packaging standardisé garantissant la **reproductibilité** [14, 15]. Un projet MLflow est défini par un fichier **`MLproject`** qui décrit les dépendances (ex. via `conda.yaml` ou `requirements.txt`) et les points d'entrée d'exécution [15].

3. **MLflow Models :**
   Convention de nommage et format d'empaquetage standardisé ("flavors") pour exporter un modèle vers divers environnements de déploiement (Docker, Kubernetes, Spark Batch Scoring, REST API, AWS SageMaker, Azure ML) [16].
   * Chaque modèle MLflow inclut un fichier de métadonnées `MLmodel` décrivant les *flavors* (ex. `python_function`, `sklearn`) [17].

4. **MLflow Model Registry :**
   Registre centralisé offrant une **gouvernance complète** sur le cycle de vie des modèles [17].
   * Permet de gérer le versionnement (v1, v2, v3...) et les transitions d'états :
     * **Experimental / Dev :** Phase de recherche [17].
     * **Staging :** Modèle en cours de validation/tests pré-production [17].
     * **Production :** Modèle actuellement déployé et servant les utilisateurs [17].
     * **Archived :** Ancien modèle remplacé [17].

---

### **D. Architecture de Stockage : Backend Store vs. Artifact Store**
Le serveur de suivi (**MLflow Tracking Server**) sépare la gestion des données en deux couches distinctes [10, 12] :
1. **Backend Store (Métadonnées) :** Stocke les identifiants de runs, les paramètres, les métriques et les états [10, 12].
   * *Mode par défaut :* Fichiers plats dans le répertoire local `./mlruns` [31].
   * *Mode Production :* Base de données relationnelle via URI SQLAlchemy (ex. SQLite `sqlite:///mlflow.db`, PostgreSQL, MySQL) [31].
2. **Artifact Store (Fichiers lourds) :** Stocke les objets binaires, les modèles sérialisés (`model.pkl`), les fichiers de configuration et les graphiques [10, 12, 17].
   * *Exemples :* Dossier local, bucket Cloud Storage [10, 32].

---

## 2. Le TP : Guide Pratique Pas-à-Pas (Atelier 5)

### **Prérequis**
* Avoir validé l'**Atelier 2** (`model_pipeline.py`), l'**Atelier 3** (`pipeline_prefect.py`) et l'**Atelier 4** (`app.py`) [28].
* Environnement virtuel WSL/Ubuntu activé [30].

---

### **Étape 1 : Préparer l'environnement et installer MLflow**

1. Activez votre environnement virtuel Python sous WSL [30] :
   ```bash
   source venv/bin/activate
   ```
2. Installez MLflow avec les dépendances déclarées dans `requirements.txt` [18, 25, 30] :
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Démarrez le serveur d'arrière-plan **Prefect** (si ce n'est pas déjà fait) [30] :
   ```bash
   prefect server start &
   ```

---

### **Étape 2 : Lancer le serveur MLflow UI**

MLflow met à disposition une interface web permettant de visualiser l'ensemble des expériences [18, 30].

1. **Lancement avec backend SQLite (Recommandé) [31] :**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
   ```
2. Accédez à l'interface graphique dans votre navigateur sur le port 5000 (`localhost:5000`) [19, 30].

---

### **Étape 3 : Intégrer MLflow dans le script `model_pipeline.py`**

Modifiez votre fichier `model_pipeline.py` pour encapsuler l'entraînement et l'évaluation dans un bloc d'expérimentation MLflow (`mlflow.start_run()`) [19, 31].

```python
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Configuration du tracking backend SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn_Prediction_Experiment")

def prepare_data(data_path="data/Churn_Modelling.csv"):
    """Charge et nettoie les données Churn."""
    df = pd.read_csv(data_path)
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    
    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_and_log_model(n_estimators=100, max_depth=10, data_path="data/Churn_Modelling.csv"):
    """Entraîne le modèle et enregistre les paramètres, métriques et artefacts."""
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    with mlflow.start_run(run_name="RandomForest_Run"):
        # 1. Logging des hyperparamètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # 2. Entraînement du modèle
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        # 3. Calcul et enregistrement des métriques
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 4. Enregistrement dans le Model Registry
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="random_forest_model",
            registered_model_name="ChurnRandomForestModel"
        )
        
        # 5. Sauvegarde locale du modèle
        joblib.dump(clf, "model.joblib")
        print(f"Run réussi ! Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        return clf

if __name__ == "__main__":
    train_and_log_model(n_estimators=100, max_depth=10)
```

---

### **Étape 4 : Adapter l'Orchestrateur Prefect (`pipeline_prefect.py`)**

Assurez-vous que votre tâche Prefect d'entraînement fait appel à la fonction enregistrant le run dans MLflow [31] :

```python
from prefect import task, flow
from model_pipeline import train_and_log_model

@task(name="Train and Log with MLflow", retries=1)
def task_train_mlflow(n_estimators=100, max_depth=10):
    return train_and_log_model(n_estimators=n_estimators, max_depth=max_depth)

@flow(name="mlflow-training-pipeline")
def flow_train_mlflow():
    task_train_mlflow(n_estimators=150, max_depth=12)

if __name__ == "__main__":
    flow_train_mlflow()
```

Exécutez le pipeline depuis le terminal [31] :
```bash
python pipeline_prefect.py
```

---

### **Étape 5 : Exploration de l'interface MLflow UI**

1. Ouvrez votre navigateur sur `localhost:5000` [19, 30].
2. **Page "Experiments" [13, 18, 31] :**
   * Sélectionnez l'expérience **`Churn_Prediction_Experiment`**.
   * Observez vos exécutions (*runs*) avec leurs métriques [13].
   * Cochez deux exécutions et cliquez sur **"Compare"** [13].
3. **Détail d'un Run & Artifacts [13, 14, 17] :**
   * Explorez l'onglet **Artifacts** (`MLmodel`, `model.pkl`, `requirements.txt`) [14, 17].
4. **Onglet "Models" (Model Registry) [17, 18] :**
   * Visualisez le modèle enregistré **`ChurnRandomForestModel`** [17, 18].
   * Effectuez la transition de la version du modèle vers l'état **`Staging`** ou **`Production`** [17].

---

### **Livrables de l'Atelier 5**
1. **Fichier `model_pipeline.py` instrumenté :** Enregistre automatiquement les hyperparamètres, les métriques et les artefacts à chaque entraînement [28, 32].
2. **Base de données backend (`mlflow.db`) :** Stocke l'historique complet des expérimentations [31].
3. **Registre de Modèles fonctionnel :** Modèle enregistré sous `ChurnRandomForestModel` dans le **MLflow Model Registry** [17, 32].

---

🐳 **Une fois le versioning avec MLflow maîtrisé, souhaitez-vous passer à l'Atelier 6 pour conteneuriser votre application FastAPI et vos artefacts MLflow avec Docker ?**
