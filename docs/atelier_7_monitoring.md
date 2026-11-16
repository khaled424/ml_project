# Séance 07 / Atelier 7 : Supervision Continue (Monitoring) MLOps avec la Stack ELK et MLflow

---

## **Partie 1 : Le Cours — Supervision et Monitoring en MLOps**

### **1. Pourquoi le Monitoring MLOps est indispensable ?**
Dans le génie logiciel traditionnel (*DevOps*), si un code fonctionne correctement lors de son déploiement en production, il continuera à fonctionner de manière déterministe (hors bugs informatiques externes ou pannes matérielles).

En **Machine Learning**, la réalité est inversée : **un modèle commence à se dégrader dès la seconde où il est mis en production**.
Le monde réel est dynamique : les comportements des utilisateurs évoluent, l'environnement socio-économique change, et les données d'entrée dérivent. Sans un monitoring continu, l'entreprise s'expose à une **dégradation silencieuse** (*silent degradation*) qui altère les décisions métier.

---

### **2. Les Principaux Phénomènes de Dégradation**

1. **Concept Drift (Dérive Conceptuelle) :**
   * **Définition :** La relation statistique entre les variables d'entrée (X) et la variable cible (y) change au fil du temps.
   * **Exemple :** Après une crise économique, un profil client autrefois considéré à faible risque de défaut de paiement peut subitement devenir à haut risque.
2. **Data Drift / Feature Drift (Dérive des Données) :**
   * **Définition :** La distribution statistique des données d'entrée en production évolue par rapport aux données d'entraînement.
   * **Exemple :** L'arrivée d'une nouvelle tranche d'âge d'utilisateurs ou l'utilisation massive de nouveaux terminaux mobiles.
3. **Dégradation Logicielle / Système (Software Health) :**
   * Consommation mémoire (RAM) excessive due à un modèle trop lourd.
   * Augmentation de la latence de réponse de l'API HTTP `/predict`.
   * Crashs de serveurs, erreurs 500 ou saturation du processeur (CPU).

---

### **3. Les 3 Piliers du Monitoring MLOps**

Le monitoring d'une application ML ne se limite pas aux seules métriques de l'algorithme ; il englobe toute la chaîne d'infrastructure.

| Pilier | Focus | Métriques Clés | Outils Recommandés |
| :--- | :--- | :--- | :--- |
| **Pilier 1 : Santé de l'Infrastructure (IT)** | Disponibilité des serveurs et des conteneurs Docker | Consommation CPU, utilisation RAM, I/O Réseau, espace disque | Metricbeat, Prometheus + Grafana |
| **Pilier 2 : Santé du Service (API Serving)** | Performances de l'API web de prédiction (FastAPI/Uvicorn) | Latence (ms), Taux d'erreur HTTP (4xx/5xx), Débit (Throughput, req/s) | Filebeat, Uvicorn logs, MLflow |
| **Pilier 3 : Santé du Modèle (Business & ML)** | Pertinence et qualité des prédictions en temps réel | Distribution des prédictions (ex. % Churn), détection des dérives, accuracy/recall en production | Elasticsearch, Kibana, MLflow, Evidently AI |

---

### **4. Architecture et Workflow de la Solution ELK**

La stack **ELK** (Elasticsearch, Logstash, Kibana), couplée aux agents **Beats** et à **MLflow**, offre une plateforme centralisée de surveillance :

```text
[ Sources (Docker Cluster) ]
 ├── FastAPI (Logs JSON, Latence, Prédictions)
 ├── MLflow (Métriques & Modèles)
 └── Docker Engine (Métriques système & conteneurs)
          │
          ▼ [ Collection Layer ]
 ├── Metricbeat (CPU, RAM, Disque)
 ├── Filebeat (Logs conteneurs)
          │
          ▼
    [ Logstash ] ──▶ (Filtrage, parsing JSON & enrichissement)
          │
          ▼
   [ Elasticsearch ] ──▶ (Indexation et stockage temps réel)
          │
          ▼
     [ Kibana ] ──▶ (Tableaux de bord & visualisations interactives)
```

---

## **Partie 2 : Le TP — Guide Pratique Pas-à-Pas**

### **Prérequis**
* Environnement **WSL (Ubuntu)** fonctionnel.
* **Docker Desktop** accessible depuis WSL et Docker Compose opérationnel.
* **Python** avec l'environnement virtuel du projet.

---

### **Étape 1 : Préparation de l'environnement et installation des packages**

Dans votre terminal WSL, depuis la racine du projet, activez l'environnement virtuel et installez les dépendances du manifeste unique :

```bash
# Activer l'environnement virtuel Python
source venv/bin/activate

# Installer les dépendances du projet et de l'atelier 7
python -m pip install -r requirements.txt
```

---

### **Étape 2 : Déploiement de la Stack ELK (Elasticsearch & Kibana)**

Le projet possède déjà un fichier `compose.yaml` qui lance l'API, Streamlit et MLflow. Ne le remplacez pas : ajoutez les deux services suivants sous sa clé `services`. Avant le premier lancement d'Elasticsearch, ajustez si nécessaire la mémoire virtuelle du noyau Linux :

```bash
sudo sysctl -w vm.max_map_count=262144
```

```yaml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.9
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "127.0.0.1:9200:9200"

  kibana:
    image: docker.elastic.co/kibana/kibana:7.17.9
    ports:
      - "127.0.0.1:5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
```

Lancez les services requis pour cet atelier :

```bash
docker compose up -d mlflow elasticsearch kibana
docker compose ps
```

---

### **Étape 3 : Script d'intégration Python (MLflow + Elasticsearch)**

Créez le script Python `monitoring_pipeline.py` pour exporter simultanément les métriques d'entraînement et de prédiction vers **MLflow** et **Elasticsearch** :

```python
import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch

# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

def run_monitored_pipeline():
    # Chargement et préparation des données
    data = pd.read_csv("data/Churn_Modelling.csv")
    X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited', 'Geography', 'Gender'])
    y = data['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # L'URI par défaut pointe vers le service MLflow du compose.yaml du projet.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))

    # Exécution MLflow avec export Elasticsearch
    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 10
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Enregistrement dans MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
    # Envoi des métriques vers Elasticsearch après la fin du run MLflow
    doc = {
        "@timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": "FINISHED",
        "metric_test_accuracy": float(acc),
        "param_n_estimators": str(n_estimators),
        "param_max_depth": str(max_depth)
    }

    res = es.index(index="mlflow-metrics", body=doc)
    print(f"[Elasticsearch] Document indexé avec ID : {res['_id']}")

if __name__ == "__main__":
    run_monitored_pipeline()
```

Exécutez le script depuis la racine du projet, avec l'environnement virtuel activé et les conteneurs MLflow et Elasticsearch démarrés :

```bash
python monitoring_pipeline.py
```

---

### **Étape 4 : Définition du Mapping Elasticsearch et Configuration Kibana**

1. Accédez à l'interface **Kibana** sur **`http://localhost:5601`**.
2. Dans **DevTools**, exécutez le script d'initialisation de l'index `PUT /mlflow-metrics` :

```json
PUT /mlflow-metrics
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "@timestamp": { "type": "date" },
      "run_id": { "type": "keyword" },
      "experiment_id": { "type": "keyword" },
      "status": { "type": "keyword" },
      "metric_test_accuracy": { "type": "float" },
      "param_n_estimators": { "type": "keyword" },
      "param_max_depth": { "type": "keyword" }
    }
  }
}
```

3. Rendez-vous dans **Stack Management > Index Patterns** et créez le pattern `mlflow-metrics` avec le filtre `@timestamp`.
4. Explorez les prédictions et métriques via l'onglet **Discover** et construisez vos tableaux de bord interactifs.

---

### **Étape 5 : Piste d'Excellence (Monitoring Système & Logs)**

Pour aller plus loin dans l'ingénierie MLOps :
* **Metricbeat :** Collecte en temps réel l'utilisation CPU et RAM des conteneurs Docker.
* **Filebeat :** Récolte les logs JSON générés par l'API FastAPI.
* **Logstash :** Filtre, enrichit et transforme les événements avant indexation dans Elasticsearch.

---

### **Livrables attendus de l'Atelier 7**
* Services **Elasticsearch** et **Kibana** ajoutés au `compose.yaml` du projet.
* **`monitoring_pipeline.py`** : Script Python couplé aux SDK `mlflow` et `elasticsearch`.
* **Index `mlflow-metrics` & Dashboard Kibana** : Dashboard de suivi des performances en temps réel sur le port `5601`.
