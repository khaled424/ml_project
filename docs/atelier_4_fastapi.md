# Atelier 4 : Exposition des fonctions via FastAPI (Cours & TP MLOps)

---

## 1. Le Cours : FastAPI dans l'Écosystème MLOps

### A. Définition et rôle de FastAPI en MLOps
En MLOps, l'étape de **Mise en Production (Serving)** consiste à rendre les prédictions d'un modèle d'apprentissage automatique accessibles aux applications tierces (web, mobile, systèmes embarqués). **FastAPI** est un framework web Python moderne, léger et hautement performant, conçu spécifiquement pour créer des APIs REST rapidement.

* **Architecture ASGI (Asynchronous Server Gateway Interface) :** FastAPI s'appuie sur **Uvicorn**, un serveur web ASGI ultra-rapide capable de gérer des requêtes asynchrones avec une faible latence.
* **Validation de données avec Pydantic :** FastAPI utilise les types de données Python standard et Pydantic pour valider automatiquement la structure des données transmises dans les requêtes JSON.
* **Documentation interactive automatique :** Il génère nativement une interface Swagger UI (accessible sur `/docs`) et ReDoc (accessible sur `/redoc`) basées sur la norme OpenAPI.

### B. Comparatif des Frameworks Web en Python

| Caractéristique | **FastAPI** | **Flask** | **Django REST Framework** | **Tornado** |
| :--- | :--- | :--- | :--- | :--- |
| **Performance** | **Très rapide** (ASGI) | Relativement lent (WSGI) | Modérée | Très rapide |
| **Validation des données** | **Automatique** (Pydantic) | Manuelle / extensions | Via Sérialiseurs | Manuelle |
| **Documentation Swagger/OpenAPI** | **Négative & Automatique** | Nécessite des extensions | Extensions requis | Non intégrée |
| **Asynchronisme native (`async/await`)** | **Oui** | Non (synchrone par défaut) | Partiel | Oui |
| **Usage recommandé MLOps** | **Idéal pour la mise en production ML** | Prototypage rapide | Applications monolithiques | Systèmes temps réel complexes |

---

## 2. Le TP : Guide pratique pas-à-pas (Atelier 4)

**Objectif du TP :** Charger le modèle entraîné et sérialisé (`model.joblib`) issu de l'Atelier 2 et l'exposer via une API REST sécurisée et documentée avec FastAPI.

---

### **Étape 1 : Préparation de l'environnement et des dépendances**

1. Activez votre environnement virtuel WSL/Ubuntu :
   ```bash
   source venv/bin/activate
   ```

2. Installez les dépendances du projet depuis le manifeste unique :
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Vérifiez la présence du fichier modèle `model.joblib` dans votre répertoire de projet `~/ml_project/`.

---

### **Étape 2 : Création de l'application API REST (`app.py`)**

Créez le fichier `app.py` dans la racine de votre projet. Ce fichier charge le modèle au démarrage et définit les schémas de données et les routes HTTP.

```python
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Prédiction du Churn Client",
    description="Service REST MLOps pour prédire le départ des clients bancaires",
    version="1.0.0"
)

# 2. Chargement du modèle entraîné
MODEL_PATH = "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print(f" Modèle chargé avec succès depuis {MODEL_PATH}")
except Exception as e:
    print(f" Erreur lors du chargement du modèle : {e}")
    model = None

# 3. Définition du schéma Pydantic pour la validation des données d'entrée
class ChurnFeatures(BaseModel):
    CreditScore: int = Field(..., example=619)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., example=42)
    Tenure: int = Field(..., example=2)
    Balance: float = Field(..., example=0.0)
    NumOfProducts: int = Field(..., example=1)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=101348.88)

# 4. Route de vérification de l'état de l'API (Healthcheck)
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Bienvenue sur l'API MLOps de Prédiction du Churn"
    }

# 5. Route POST de prédiction
@app.post("/predict")
def predict_churn(features: ChurnFeatures):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Le modèle ML n'est pas chargé sur le serveur."
        )
    
    try:
        # Convertir les données Pydantic en DataFrame Pandas
        input_data = pd.DataFrame([features.dict()])
        
        # Réaliser la prédiction
        prediction = model.predict(input_data)
        prediction_value = int(prediction[0])
        
        # Calcul des probabilités si disponible
        proba = None
        if hasattr(model, "predict_proba"):
            proba_array = model.predict_proba(input_data)
            proba = float(proba_array[0][1])  # Probabilité d'appartenir à la classe 1 (Exited)

        return {
            "prediction": prediction_value,
            "interpretation": "Client à risque de départ (Churn)" if prediction_value == 1 else "Client fidèle",
            "probability_churn": proba
        }
    except Exception as err:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur lors du calcul de la prédiction : {str(err)}"
        )
```

---

### **Étape 3 : Démarrage du serveur et tests interactifs**

#### **1. Démarrer le serveur Uvicorn en mode développement**
Dans votre terminal WSL, lancez la commande suivante :
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
* `--reload` : Recharge automatiquement le serveur dès qu'une modification est enregistrée dans `app.py`.
* `--host 0.0.0.0` : Rend l'API accessible sur toutes les interfaces réseau.

*(Optionnel - Démarrage en arrière-plan sans bloquer le terminal)* :
```bash
nohup uvicorn app:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

#### **2. Tester via Swagger UI (Interface interactive)**
1. Ouvrez votre navigateur web et rendez-vous sur : `http://127.0.0.1:8000/docs`
2. Cliquez sur le point de terminaison **`POST /predict`**.
3. Cliquez sur **"Try it out"**.
4. Remplissez le corps de la requête JSON avec des données de test et cliquez sur **"Execute"**.
5. Observez le code de réponse HTTP (`200 OK`) et le JSON de retour contenant la prédiction.

---

### **Étape 4 : Intégration de l'API dans le pipeline Prefect (`pipeline_prefect.py`)**

Pour automatiser le lancement de l'API web au sein de vos workflows d'orchestration Prefect, ajoutez la tâche et le flux suivants dans votre fichier `pipeline_prefect.py` :

```python
import subprocess
from prefect import task, flow

@task(name="Start FastAPI Server")
def task_start_api():
    print("🚀 Démarrage du serveur FastAPI via Uvicorn...")
    # Lancement du serveur Uvicorn
    cmd = ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    process = subprocess.Popen(cmd)
    print(f" API démarrée avec le PID: {process.pid}")
    return process.pid

@flow(name="api-flow")
def flow_api():
    task_start_api()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--flow" and sys.argv[2] == "api":
        flow_api()
```

Vous pouvez désormais démarrer l'API directement depuis la ligne de commande Prefect :
```bash
python pipeline_prefect.py --flow api
```

---

## 3. Synthèse et Livrables de l'Atelier 4

* **`app.py`** : Code source complet définissant le service web REST, les schémas Pydantic et l'exposition de la fonction `predict()`.
* **Documentation Swagger UI** : Accessible en local sur `http://127.0.0.1:8000/docs` pour tester interactivement le modèle.
* **Intégration Prefect** : Intégration du flux `flow_api()` pour orchestrer l'exécution du service de prédiction.
