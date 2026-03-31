# 👤 Face Detect Tool

## 📝 Description
Ce projet est un outil de reconnaissance faciale capable de traiter des fichiers images (potentiellement issus d'un `.zip`), d'extraire les visages via **OpenCV**, de générer des embeddings et de les stocker dans **Milvus** (base de données vectorielle) pour identification. L'interface utilisateur est propulsée par **Streamlit**.

---

## 📂 Arborescence du Projet
Voici la structure principale pour t'aider à naviguer :

```text
.
├── data/                   # Données de l'application
│   ├── images/             # Images sources et extractions
│   │   └── faces/          # Visages détectés et recadrés
│   └── face_model.xml      # Modèle Haar Cascade (généré par make)
├── db/                     # Configuration de la base de données
│   └── docker-compose.yml  # Stack Milvus Standalone
├── ml/                     # Logique Machine Learning
│   ├── model/              # Modèles de reconnaissance (facenet, etc.)
│   ├── embedding.py        # Génération des vecteurs
│   └── search.py           # Logique de recherche vectorielle
├── view/                   # Interface Utilisateur
│   ├── display.py          # Point d'entrée Streamlit
│   └── interface.py        # Composants UI
├── docker-compose.yml      # Stack de l'application principale
├── Dockerfile              # Containerisation de l'App (Python + OpenCV)
├── Makefile                # Automatisation des tâches (Indispensable)
├── pyproject.toml          # Gestion des dépendances avec UV
└── .env                    # Variables d'environnement (généré par make)
```

---

## 🛠 Prérequis
Le projet utilise des outils modernes pour garantir la performance et l'isolation :
* **[uv](https://github.com/astral-sh/uv)** : Pour la gestion ultra-rapide des dépendances Python.
* **Docker & Docker Compose** : Pour l'exécution des services (App + Milvus).
* **Make** : Pour piloter le projet simplement.

---

## Installation et Lancement Rapide

### 1. Récupération du package
Récupérer l'image de la dernière version de l'app: 
**[App Image](https://github.com/Lukombo-JDS/face-detect-laforge/pkgs/container/face-detect-laforge)**

### 2. Lancer le container
Lancer le container 
```bash
make app-up
```

L'app se lance à cette adresse en local [APP adresse](http://localhost:8501)

##  Installation et Lancement Manuel

### 1. Initialisation locale (Développement)
Pour préparer ton environnement de travail sans Docker :
```bash
make all
```
*Cette commande : crée l'arborescence, génère le `.env`, synchronise les dépendances via `uv sync` et télécharge le modèle OpenCV.*

### 2. Lancement des services
Le projet est divisé en deux parties (App et DB) :

**Lancer la base de données (Milvus) :**
```bash
make db-up
```

**Lancer l'application en mode Debug (Streamlit local) :**
```bash
make run-debug
```

---

## 🐳 Docker & Déploiement

### Utilisation via le Registre (GHCR)
Si tu souhaites récupérer l'image déjà buildée par l'équipe :
```bash
make deploy-from-registry
```

### Build et Push (si tu fais des modifs)
Pour mettre à jour l'image sur le GitHub Container Registry :
1.  **Build & Push** : `make build-and-push`
2.  **Nettoyage** : `make clr-all-img` (pour vider les données de test avant un build propre).

---

## 🛠 Commandes utiles (Makefile)

| Commande | Description |
| :--- | :--- |
| `make help` | Affiche la liste de toutes les commandes disponibles. |
| `make venv` | Synchronise l'environnement virtuel avec `pyproject.toml`. |
| `make download` | Récupère les fichiers XML de détection OpenCV. |
| `make clr-all-img` | Vide récursivement les dossiers `data/images/`. |
| `make db-down` | Arrête proprement la stack Milvus. |

---

## 🌐 Accès à l'Application
Une fois le conteneur lancé (via Docker ou `make run-debug`), l'interface Streamlit est accessible à l'adresse suivante :

👉 **URL :** [http://localhost:8501](http://localhost:8501)

---

## 🔑 Accès à l'Image de "Release" (GitHub Packages)
L'image officielle du projet est stockée sur le **GitHub Container Registry (GHCR)**. Pour l'utiliser, un accès en lecture est nécessaire.

---

## 💡 Notes techniques
* **OpenCV** : Le projet utilise `haarcascade_frontalface_default.xml` pour la détection. Il est stocké dans `./data/`.
* **Variables d'environnement** : Le fichier `.env` définit `OUTPUT_IMAGES_DIR`. Assure-toi qu'il correspond à ton montage de volume Docker si tu modifies la structure.
* **Optimisation Docker** : Un fichier `.dockerignore` est présent pour éviter d'embarquer la `.venv` locale ou les images de test dans l'image finale.

---
*Dernière mise à jour : Mars 2026*
