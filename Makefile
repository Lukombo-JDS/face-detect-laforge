# Variables
UV = uv
PYTHON_VERSION = 3.12
DATA_DIR = ./data
IMG_DIR = $(DATA_DIR)/images
FACES_DIR = $(IMG_DIR)/faces
HAARCASCADE_URL = https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
HAARCASCADE_FILE = $(DATA_DIR)/haarcascade_frontalface_default.xml
ENV_FILE = .env

.PHONY: all venv download docker-up docker-down init env clr-all-img run-debug help

all: init env venv download ## Configuration complète (dossiers + env + venv + download)

## --- CONFIGURATION & DOSSIERS ---

init: ## Crée la structure des dossiers de données et d'images
	@echo "--- Création de l'arborescence ---"
	@mkdir -p $(FACES_DIR)
	@echo "Dossiers créés dans $(DATA_DIR)"

env: ## Crée le fichier .env avec les variables par défaut
	@echo "--- Génération du fichier $(ENV_FILE) ---"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo 'OUTPUT_IMAGES_DIR = "data/images"' > $(ENV_FILE); \
		echo "Fichier $(ENV_FILE) créé."; \
	else \
		echo "Le fichier $(ENV_FILE) existe déjà."; \
	fi

## --- ENVIRONNEMENT PYTHON ---

venv: ## Crée l'environnement virtuel avec uv et installe les dépendances
	@echo "--- Configuration de l'environnement Python ---"
	$(UV) venv --python $(PYTHON_VERSION)
	$(UV) pip install -r requirements.txt

download: ## Télécharge le modèle Haar Cascade pour OpenCV
	@echo "--- Téléchargement du modèle Haar Cascade ---"
	@mkdir -p $(DATA_DIR)
	@if [ ! -f $(HAARCASCADE_FILE) ]; then \
		curl -L $(HAARCASCADE_URL) -o $(HAARCASCADE_FILE); \
		echo "Fichier téléchargé : $(HAARCASCADE_FILE)"; \
	else \
		echo "Le fichier cascade existe déjà."; \
	fi

## --- DÉVELOPPEMENT & EXECUTION ---

run-debug: ## Lance l'interface Streamit en mode debug via uv
	@echo "--- Lancement de Streamlit ---"
	$(UV) run streamlit run view/display.py

clr-all-img: ## Supprime toutes les images (faces et raw)
	@echo "--- Nettoyage des images ---"
	@rm -rf $(FACES_DIR)/*
	@rm -f $(IMG_DIR)/*.jpg
	@echo "Images cleared"

## --- DOCKER ---

docker-up: ## Lance Docker Compose (build + detach)
	@echo "--- Lancement de Docker Compose ---"
	docker-compose up -d --build

docker-down: ## Arrête et supprime les conteneurs Docker
	docker-compose down

## --- AIDE ---

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
