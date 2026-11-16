# Pipeline de prédiction du churn client

Ce projet charge `Churn_Modelling.csv`, prépare les données, entraîne une
Random Forest, affiche les métriques et sauvegarde le modèle dans
`classifier.joblib`.

## 1. Ouvrir le projet dans WSL

Dans le terminal WSL :

```bash
cd ~/ml_project
```

Le fichier CSV doit être dans ce dossier. Vérifiez sa présence avec :

```bash
ls
```

## 2. Activer l'environnement Python

```bash
source venv/bin/activate
python --version
```

Quand l'environnement est activé, l'invite commence généralement par `(venv)`.
La commande correcte est `python`, et non `pyhton` (qui explique le message
`command not found`). Si la commande `python` n'est pas disponible dans WSL,
essayez `python3`, ou recréez l'environnement avec `python3 -m venv venv`.

Si les dépendances ne sont pas encore installées :

```bash
python -m pip install -r requirements.txt
```

Pour exécuter les tests automatisés, installez aussi pytest si nécessaire :

```bash
python -m pip install pytest
```

## 3. Tester le pipeline complet manuellement

Depuis `~/ml_project`, avec `(venv)` activé :

```bash
python main.py
```

Le programme doit afficher l'accuracy, le rapport de classification et la
matrice de confusion. Le modèle est écrit dans `classifier.joblib`.

Vous pouvez modifier les chemins si besoin :

```bash
python main.py --data-path ./Churn_Modelling.csv --model-path ./classifier.joblib
```

Le script accepte aussi les options `--prepare`, `--train` et `--evaluate`,
utilisées par le flow Prefect. Exécute-les dans cet ordre :

```bash
python main.py --prepare
python main.py --train
python main.py --evaluate
```

`--prepare` crée `prepared_data.joblib`, `--train` lit ce fichier et crée
`classifier.joblib`, puis `--evaluate` recharge ces deux artefacts pour afficher
les métriques. Si tu modifies les chemins, utilise les mêmes options à chaque
étape :

```bash
python main.py --prepare --data-path ./Churn_Modelling.csv --prepared-data-path ./prepared_data.joblib
python main.py --train --prepared-data-path ./prepared_data.joblib --model-path ./classifier.joblib
python main.py --evaluate --prepared-data-path ./prepared_data.joblib --model-path ./classifier.joblib
```

Les étapes `--train` et `--evaluate` dépendent des fichiers créés auparavant :
exécuter `--train` seul échoue si `prepared_data.joblib` n'existe pas encore ;
exécuter `--evaluate` exige à la fois les données préparées et un modèle
entraîné. Sans option d'étape, `python main.py` continue d'exécuter tout le
pipeline en une fois.

## 4. Tester les fonctions individuellement

Ouvrez l'interpréteur Python depuis le dossier du projet :

```bash
python
```

Puis exécutez les étapes ci-dessous dans le même interpréteur :

```python
from model_pipeline import (
    load_data, preprocess_data, split_data, train_model,
    evaluate_model, save_model, load_model
)

# Chargement : retourne un DataFrame.
df = load_data("Churn_Modelling.csv")
print(df.shape, df.columns.tolist())

# Prétraitement : retourne X (variables) et y (cible).
X, y = preprocess_data(df)
print(X.shape, y.shape, X.dtypes)

# Séparation : retourne X_train, X_test, y_train, y_test.
X_train, X_test, y_train, y_test = split_data(X, y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Entraînement : retourne un modèle ajusté.
model = train_model(X_train, y_train)
print(type(model).__name__)

# Évaluation : retourne les métriques et les prédictions.
metrics = evaluate_model(model, X_test, y_test)
print("Accuracy:", metrics["accuracy"])
print(metrics["classification_report"])
print(metrics["confusion_matrix"])

# Sauvegarde puis rechargement du modèle.
save_model(model, "classifier.joblib")
loaded_model = load_model("classifier.joblib")
print(loaded_model.predict(X_test.head()))
```

Quittez l'interpréteur avec `exit()`.

Chaque fonction vérifie ses entrées et documente son résultat ainsi que les
erreurs possibles dans sa docstring. Par exemple, le prétraitement signale la
colonne cible manquante, des valeurs manquantes ou des variables explicatives
non numériques.

## 5. Lancer les tests unitaires

Depuis le dossier du projet, toujours avec l'environnement activé :

```bash
python -m pytest -v test_model_pipeline.py
```

Les tests vérifient les principales fonctions avec un petit jeu de données de
test temporaire. Ils ne modifient pas le CSV de production ni le modèle existant.

## 6. Fiche de révision : explication du TP

### Prérequis et objectifs

Le notebook `customer_churn.ipynb` fonctionne et contient les étapes
d'exploration, de préparation, d'entraînement et d'évaluation. Le but du TP est
de transformer ces cellules en fonctions réutilisables, puis de les exécuter
depuis un script principal.

Les deux livrables sont :

- `model_pipeline.py` : fonctions de préparation, entraînement, évaluation et
  sérialisation du modèle.
- `main.py` : point d'entrée qui orchestre les fonctions du pipeline.

La fonction `prepare_data()` combine `load_data()`, `preprocess_data()` et
`split_data()` pour réaliser toute la préparation et créer les ensembles
d'entraînement et de test.

### Étape 1 — Identifier les blocs logiques du notebook

| Bloc logique | Fonction correspondante |
|---|---|
| Lecture du CSV | `load_data()` |
| Encodage et suppression des colonnes inutiles | `preprocess_data()` |
| Préparation complète et séparation train/test | `prepare_data()` (qui appelle les fonctions ci-dessus et `split_data()`) |
| Entraînement du classifieur | `train_model()` |
| Calcul des métriques et de la matrice de confusion | `evaluate_model()` |
| Écriture et lecture du modèle avec joblib | `save_model()` et `load_model()` |

Le modèle effectivement utilisé est `RandomForestClassifier`. Le notebook
contenait aussi des imports et des cellules d'exploration qui ne sont pas
nécessaires pour exécuter le pipeline final.

### Étape 2 — Charger et préparer les données

#### `load_data(data_path)`

- **Entrée :** chemin vers le fichier CSV, par exemple `Churn_Modelling.csv`.
- **Sortie :** un DataFrame pandas.
- **Vérifications :** le chemin doit exister et le CSV ne doit pas être vide.

Le chargement de ce projet donne un DataFrame de forme `(10000, 14)` :
10 000 lignes et 14 colonnes. Une ligne correspond à un client.

#### `preprocess_data(df)`

- **Entrée :** le DataFrame brut.
- **Sortie :** le tuple `(X, y)`.
- `X` contient les variables explicatives fournies au modèle.
- `y` contient la cible à prédire, la colonne `Exited`.

La fonction encode `Gender`, puis retire les colonnes `Surname`, `Geography`,
`RowNumber` et `CustomerId`. Les neuf variables explicatives restantes sont :
`CreditScore`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`,
`HasCrCard`, `IsActiveMember` et `EstimatedSalary`.

Dans l'exécution observée, `X` a la forme `(10000, 9)` et `y` la forme
`(10000,)`. La fonction vérifie notamment que le DataFrame et la cible sont
présents, qu'il n'y a pas de valeurs manquantes et que les variables
explicatives sont numériques.

**Choix à connaître :** `Geography` est supprimée ; le modèle n'apprend donc
pas d'effet lié au pays. C'est une limite du jeu de variables retenu.

#### `split_data(X, y)`

La fonction sépare les données avec `test_size=0.2` :

- `X_train`, `y_train` : données utilisées pour apprendre, 8 000 lignes ;
- `X_test`, `y_test` : données réservées à l'évaluation, 2 000 lignes.

`test_size=0.2` réserve 20 % des observations au test. `random_state=1` rend
la séparation reproductible. `stratify=y` conserve approximativement la même
proportion de chaque classe dans les deux ensembles.

### Étape 3 — Entraîner le modèle

#### `train_model(X_train, y_train)`

La fonction construit un `RandomForestClassifier` avec 100 arbres
(`n_estimators=100`) et `random_state=42`, puis l'entraîne sur les données
reçues. Elle retourne le modèle ajusté.

L'affichage `RandomForestClassifier` confirme le type du modèle retourné.
La forêt aléatoire n'exige pas de mise à l'échelle des variables pour ce
pipeline ; les variables numériques sont utilisées directement.

### Étape 4 — Évaluer les performances

#### `evaluate_model(model, X_test, y_test)`

La fonction prédit la classe sur le jeu de test et retourne un dictionnaire
contenant :

- `accuracy` : proportion totale de prédictions correctes ;
- `predictions` : classes prédites ;
- `classification_report` : precision, recall, F1-score et nombre d'exemples ;
- `confusion_matrix` : détail des bonnes et mauvaises prédictions.

#### Accuracy obtenue

L'exécution donne une accuracy de **85,25 %**, soit 1 705 prédictions
correctes sur 2 000 :

```text
(1532 + 173) / 2000 = 0.8525 = 85.25 %
```

Cette mesure est globale. Elle ne suffit pas à elle seule pour juger le modèle,
car les deux classes n'ont pas le même nombre d'exemples.

#### Rapport de classification

Dans ce projet, `Exited=0` signifie que le client est resté et `Exited=1`
signifie qu'il est parti (churn). La colonne `support` donne le nombre de cas
réels dans le jeu de test : 1 593 clients restés et 407 clients partis.

Pour la classe 1 :

- **Precision = 0,74 :** parmi les clients prédits comme partis, environ 74 %
  sont réellement partis.
- **Recall = 0,43 :** le modèle détecte environ 43 % des clients réellement
  partis. Il en manque donc une partie importante.
- **F1-score = 0,54 :** mesure qui combine precision et recall.

Le recall de la classe 1 est particulièrement important si l'objectif métier
est de repérer les clients à risque avant leur départ. Une accuracy élevée ne
signifie pas automatiquement que le modèle détecte bien les départs.

#### Matrice de confusion

Dans la matrice scikit-learn, les lignes représentent les classes réelles et
les colonnes les classes prédites :

```text
                 Prédit 0   Prédit 1
Réel 0              1532         61
Réel 1               234        173
```

- **1 532** : clients restés correctement prédits comme restés.
- **61** : clients restés prédits à tort comme partis.
- **234** : clients partis que le modèle n'a pas détectés.
- **173** : clients partis correctement détectés.

Cette matrice explique les scores du rapport : le modèle identifie beaucoup
de clients restés, mais manque 234 des 407 clients partis.

#### Comprendre les cinq prédictions affichées

```text
[0 0 0 0 0]
```

Ce résultat signifie que le modèle prédit « client resté » pour les cinq
premières lignes de `X_test`. Ce ne sont pas nécessairement les cinq premières
lignes du CSV. Pour vérifier si ces prédictions sont correctes, compare-les
aux vraies valeurs avec `y_test.head()`.

### Étape 5 — Sauvegarder et charger le modèle

#### `save_model(model, output_path)`

La fonction sérialise le Random Forest avec joblib. Sans argument, le chemin
par défaut est `/home/khaled/ml_project/classifier.joblib`.

#### `load_model(model_path)`

La fonction recharge un modèle sauvegardé et retourne le
`RandomForestClassifier`. Il est alors possible de réutiliser le modèle sans
le réentraîner :

```python
loaded_model = load_model("classifier.joblib")
predictions = loaded_model.predict(X_test.head())
```

Ne charge que des fichiers joblib de confiance.

### Rôle de `main.py`

`main.py` appelle les étapes dans l'ordre :

```text
chargement → prétraitement → séparation → entraînement
→ évaluation → sauvegarde
```

Pour lancer le pipeline dans WSL :

```bash
cd ~/ml_project
source venv/bin/activate
python main.py
```

Les chemins d'entrée et de sortie peuvent aussi être fournis en arguments :

```bash
python main.py --data-path ./Churn_Modelling.csv --model-path ./classifier.joblib
```

La commande est `python`, et non `pyhton`.

### Tests unitaires

La commande suivante exécute les tests :

```bash
python -m pytest -v test_model_pipeline.py
```

Le résultat observé est **12 tests réussis** (`12 passed`). Ils vérifient
notamment le chargement, le prétraitement, la séparation, l'entraînement,
l'évaluation, la sauvegarde et le chargement, ainsi que des entrées invalides
(fichier absent, cible absente, valeurs manquantes, taille de test invalide).

Ces tests confirment que les fonctions respectent les comportements attendus.
Ils ne mesurent pas la qualité prédictive du modèle : celle-ci est évaluée avec
le jeu de test et les métriques ci-dessus.

### Résumé à retenir

Le notebook exploratoire a été séparé en fonctions documentées dans
`model_pipeline.py`, et `main.py` les orchestre. Le pipeline a fonctionné sur
le CSV : il obtient 85,25 % d'accuracy. Pour interpréter le résultat
correctement, il faut aussi signaler le recall de 43 % sur les clients partis
et les 234 départs manqués, au lieu de s'arrêter à l'accuracy.

## 7. Orchestrer le pipeline avec Prefect

Le fichier `pipeline_prefect.py` définit quatre flows :

- `code` : installation des dépendances, vérification Flake8 et tests pytest ;
- `train` : préparation des données puis entraînement ;
- `evaluate` : évaluation des données préparées avec le modèle sauvegardé ;
- `all` : contrôles, préparation, entraînement et évaluation.

Depuis `~/ml_project`, avec l'environnement virtuel activé :

```bash
python pipeline_prefect.py --flow code
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow evaluate
```

Le flow `evaluate` nécessite que `train` ait déjà créé les fichiers
`prepared_data.joblib` et `classifier.joblib`. Pour tout exécuter d'un coup :

```bash
python pipeline_prefect.py --flow all
```

Pour démarrer le serveur Prefect et créer un déploiement planifié, suis la
section « Automatisation et Déploiement » de `atelier_3_prefect.md`. Le script
`deploiement_prefect.py` enregistre `flow_all` avec une planification
quotidienne à 02:00, heure de Tunis.

Lance le serveur dans un premier terminal :

```bash
prefect server start
```

Dans un deuxième terminal, configure l'API et démarre le déploiement :

```bash
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
python deploiement_prefect.py
```

Garde ce terminal ouvert. Dans un troisième terminal, tu peux lister et
déclencher le déploiement :

```bash
prefect deployment ls
prefect deployment run 'all/ml-pipeline-all'
```

L'interface de suivi est disponible sur `http://localhost:4200`.
