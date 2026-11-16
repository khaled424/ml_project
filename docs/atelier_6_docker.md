# Atelier 6 & Séance 6 : Conteneurisation MLOps avec Docker et FastAPI

---

## 1. Le Cours : Conteneurisation avec Docker dans un Pipeline MLOps

### **A. Virtualisation Lourde (VM) vs Virtualisation Légère (Conteneurs)**
Dans un environnement de développement et de déploiement classique, l'un des plus grands défis est le fameux problème **"Ça marche sur ma machine !"** (*It works on my machine*). Cela est dû aux divergences d'OS, de dépendances Python, ou de configurations système entre le poste du Data Scientist et les serveurs de production.

* **Virtualisation lourde (Machines Virtuelles / VM) :** Emule un système matériel complet au-dessus d'un hyperviseur. Chaque VM contient son propre système d'exploitation invité (Guest OS), ce qui la rend lourde en mémoire, lente à démarrer et gourmande en ressources.
* **Virtualisation légère (Conteneurs Docker) :** Partage le noyau (*Kernel*) du système d'exploitation hôte. Chaque conteneur s'exécute dans un espace utilisateur isolé, ce qui garantit une vitesse de démarrage quasi instantanée, une empreinte mémoire très faible et une portabilité totale.

---

### **B. Pourquoi Docker est essentiel en MLOps ?**
En MLOps, une application ne se limite pas à du code source : elle combine le **code Python**, les **dépendances** (`requirements.txt`), la **version de Python**, l'**API web (FastAPI)** et les **artefacts de modèles (joblib / MLflow)**. 

Docker permet d'empaqueter l'intégralité de cette pile applicative dans un artefact unique, immuable et léger : **l'image Docker**.

---

### **C. Architecture et Concepts Clés de Docker**

1. **Docker Engine / Daemon :** Le service d'arrière-plan responsable de la création, du lancement et de la gestion des conteneurs et des images.
2. **Dockerfile :** Fichier texte de configuration sans extension contenant la suite d'instructions permettant de construire automatiquement une image Docker (l'analogie avec la *recette de cuisine*).
3. **Image Docker :** Modèle en lecture seule (*template*) figé contenant le code, le runtime, les bibliothèques et les variables d'environnement (le *plat préparé* d'après la recette).
4. **Conteneur Docker :** Instance active en cours d'exécution d'une image Docker. Un même conteneur peut être dupliqué à l'infini à partir d'une seule image.
5. **Docker Hub :** Registre public centralisé (*SaaS*) permettant de stocker, taguer, versionner et distribuer les images Docker au sein des équipes.

---

### **D. Les Instructions Fondamentales du Dockerfile**

| Instruction | Rôle & Description |
| :--- | :--- |
| `FROM` | Spécifie l'image de base (ex: `python:3.9-slim`). |
| `LABEL` | Ajoute des métadonnées (auteur, version, description). |
| `WORKDIR` | Définit le répertoire de travail pour toutes les instructions suivantes (`RUN`, `COPY`, `CMD`). |
| `COPY` / `ADD` | Copie des fichiers/dossiers de la machine hôte vers le système de fichiers du conteneur. |
| `RUN` | Exécute des commandes système pendant la phase de construction de l'image (ex: `pip install -r requirements.txt`). |
| `EXPOSE` | Indique le port réseau sur lequel le conteneur écoutera au runtime (ex: `8000`). |
| `ENV` | Définit des variables d'environnement dans le conteneur. |
| `CMD` / `ENTRYPOINT` | Définit la commande par défaut exécutée au démarrage du conteneur (ex: `uvicorn app:app --host 0.0.0.0 --port 8000`). |

---

### **E. Commandes Docker Indispensables (Cheat Sheet)**

```bash
# 1. Construire une image depuis le Dockerfile du dossier courant
docker build -t nom_image .

# 2. Lister les images locales disponibles
docker images

# 3. Lancer un conteneur avec mappage de port (Hôte:Conteneur) en arrière-plan (-d)
docker run -d -p 8000:8000 --name nom_conteneur nom_image

# 4. Lister les conteneurs actifs
docker ps

# 5. Lister tous les conteneurs (actifs et arrêtés)
docker ps -a

# 6. S'authentifier sur Docker Hub
docker login

# 7. Taguer une image pour Docker Hub
docker tag nom_image username_dockerhub/nom_repository:version

# 8. Pousser l'image vers Docker Hub
docker push username_dockerhub/nom_repository:version

# 9. Supprimer une image Docker
docker image rm -f nom_image
```

---

## 2. Le TP : Guide Pratique Pas-à-Pas (Atelier 6)

### **Prérequis**
* **WSL / Ubuntu** configuré.
* Service **Docker Engine** installé et actif (`docker --version`).
* Fichiers du projet prêts : `app.py` (FastAPI), `model_pipeline.py`, `requirements.txt` et `artifacts/classifier.joblib`.

---

### **Étape 1 : Créer le fichier `Dockerfile`**

Dans le répertoire racine de votre projet ML (`~/ml_project/`), créez le fichier `Dockerfile` :

```dockerfile
# 1. Image de base légère Python
FROM python:3.9-slim

# 2. Métadonnées du projet
LABEL maintainer="MLOps Student"
LABEL description="Image Docker FastAPI & MLflow Churn Prediction"

# 3. Répertoire de travail dans le conteneur
WORKDIR /app

# 4. Copie du fichier de dépendances
COPY requirements.txt ./

# 5. Installation du sous-ensemble API, avec les versions contraintes par le manifeste
RUN pip install --no-cache-dir -c requirements.txt fastapi joblib numpy pandas scikit-learn uvicorn

# 6. Copie du code source et des artefacts modèles
COPY app.py .
COPY model_pipeline.py .
COPY artifacts/classifier.joblib ./artifacts/classifier.joblib
# Si vous utilisez des artefacts MLflow : COPY mlruns/ ./mlruns/

# 7. Exposer le port de l'API FastAPI
EXPOSE 8000

# 8. Commande de démarrage de l'API avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### **Étape 2 : Construire l'image Docker (`docker build`)**

Dans le terminal WSL, exécutez la commande de construction en appliquant la convention de nommage requise par l'atelier (`prenom_nom_classe_mlops`) :

```bash
docker build -t sirine_naifar_4si_mlops .
```

Une fois le *build* terminé, vérifiez que l'image apparaît bien dans la liste locale :
```bash
docker images
```

---

### **Étape 3 : Tester l'image et le conteneur en local (`docker run`)**

1. **Lancer le conteneur Docker :**
   ```bash
   docker run -d -p 8000:8000 --name api_churn_container sirine_naifar_4si_mlops
   ```

2. **Vérifier l'état du conteneur :**
   ```bash
   docker ps
   ```

3. **Tester l'API conteneurisée :**
   * Ouvrez votre navigateur web et accédez à **`http://localhost:8000/docs`**.
   * Exécutez une requête sur le point de terminaison **`POST /predict`** avec un objet JSON de test pour valider que le modèle effectue la prédiction correctement à l'intérieur du conteneur.

4. **Consulter les logs du conteneur :**
   ```bash
   docker logs api_churn_container
   ```

---

### **Étape 4 : Publier l'image sur Docker Hub (`tag` & `push`)**

1. **Connexion à Docker Hub :**
   ```bash
   docker login
   ```
   *(Renseignez vos identifiants Docker Hub).*

2. **Taguer l'image locale avec votre nom d'utilisateur Docker Hub :**
   ```bash
   docker tag sirine_naifar_4si_mlops votre_username_dockerhub/sirine_naifar_4si_mlops:1.0
   ```

3. **Pousser l'image vers le registre public :**
   ```bash
   docker push votre_username_dockerhub/sirine_naifar_4si_mlops:1.0
   ```

4. **Vérification :** Connectez-vous sur [Docker Hub](https://hub.docker.com/) pour confirmer que l'image est bien en ligne et prête à être déployée sur n'importe quel serveur distant via `docker pull`.

---

### **Étape 5 : Automatiser la Conteneurisation dans Prefect (Pipeline CD)**

Pour intégrer le processus de déploiement continu dans **Prefect**, mettez à jour votre fichier `pipeline_prefect.py` en y ajoutant des tâches de conteneurisation :

```python
import subprocess
from prefect import task, flow

@task(name="Build Docker Image")
def build_docker_image_task(image_name: str):
    print(f"Construction de l'image Docker {image_name}...")
    cmd = f"docker build -t {image_name} ."
    subprocess.run(cmd, shell=True, check=True)
    print("Image construite avec succès !")

@task(name="Push Docker Image")
def push_docker_image_task(image_name: str, dockerhub_user: str, tag: str = "latest"):
    target_tag = f"{dockerhub_user}/{image_name}:{tag}"
    print(f"Tag de l'image vers {target_tag}...")
    subprocess.run(f"docker tag {image_name} {target_tag}", shell=True, check=True)
    print("Push vers Docker Hub...")
    subprocess.run(f"docker push {target_tag}", shell=True, check=True)
    print("Image publiée avec succès sur Docker Hub !")

@flow(name="cd-docker-pipeline")
def flow_cd():
    image_name = "sirine_naifar_4si_mlops"
    dockerhub_user = "votre_username_dockerhub"
    
    build_docker_image_task(image_name)
    push_docker_image_task(image_name, dockerhub_user)

if __name__ == "__main__":
    flow_cd()
```

Vous pouvez désormais exécuter votre pipeline CD en tapant :
```bash
python pipeline_prefect.py --flow cd
```

---

## 3. Livrables Attendus & Pistes d'Excellence

### **Livrables :**
1. **`Dockerfile`** fonctionnel et optimisé à la racine du projet.
2. **Image Docker** construite, testée localement sur le port `8000` et poussée sur Docker Hub.
3. **Pipeline Prefect (CD)** permettant d'automatiser la construction et la publication de l'image.

### **Pistes d'Excellence :**
* **Multi-Stage Build :** Réduire la taille de l'image Docker en séparant l'étape de *build* des dépendances de l'étape finale d'exécution.
* **Interface Frontend dédiée :** Créer une deuxième image Docker (ex: avec Streamlit ou HTML/JS) communiquant avec le conteneur FastAPI.
* **MLflow Tracking Server Conteneurisé :** Lancer un conteneur dédié uniquement au serveur MLflow connecté à une base SQLite persistante.
