# Variables
UV = uv
PYTHON_VERSION = 3.12
DATA_DIR = ./data
DB_DIR = ./db
IMG_DIR = $(DATA_DIR)/images
FACES_DIR = $(IMG_DIR)/faces
HAARCASCADE_URL = https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
HAARCASCADE_FILE = $(DATA_DIR)/haarcascade_frontalface_default.xml
ENV_FILE = .env
GITHUB_USER = lukombo-jds
REPO_NAME = face-detect-laforge
VERSION_BASE ?= v.0.0.1
IMAGE_ID ?= unstable-$(shell date +%Y%m%d%H%M%S)
IMAGE_VERSION ?= $(VERSION_BASE)-$(IMAGE_ID)
IMAGE_TAG = ghcr.io/$(GITHUB_USER)/$(REPO_NAME):$(IMAGE_VERSION)
IMAGE_TAG_LATEST = ghcr.io/$(GITHUB_USER)/$(REPO_NAME):latest
DOCKER_COMPOSE = docker compose
COMPOSE_APP = docker-compose.yml
COMPOSE_DB = db/docker-compose.yml

.PHONY: all venv download docker-up docker-down init env clr-all-img run-debug help check-docker docker-build docker-smoke build-and-push release-image release-unstable release-stable show-image-tag

all: init env venv download ## Configuration complète (dossiers + env + venv + download)

## --- CONFIGURATION & DOSSIERS ---

init: ## Crée la structure des dossiers de données et d'images
	@echo "--- Création de l'arborescence ---"
	@mkdir -p $(FACES_DIR)
	@echo "Dossiers créés dans $(DATA_DIR)"
	@mkdir -p $(DB_DIR)
	@echo "Dossier créé à la racine"

env: ## Crée le fichier .env avec les variables par défaut
	@echo "--- Génération du fichier $(ENV_FILE) ---"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo 'OUTPUT_IMAGES_DIR = "data/images"' > $(ENV_FILE); \
		echo "Fichier $(ENV_FILE) créé."; \
	else \
		echo "Le fichier $(ENV_FILE) existe déjà."; \
	fi

## --- ENVIRONNEMENT PYTHON ---

venv: ## Synchronise l'environnement virtuel avec pyproject.toml via uv
	@echo "--- Installation des dépendances (pyproject.toml) ---"
	$(UV) sync --python $(PYTHON_VERSION)

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

run-debug: ## Lance l'interface Streamlit en mode debug via uv
	@echo "--- Lancement de Streamlit ---"
	$(UV) run streamlit run view/display.py

clr-all-img: ## Supprime toutes les images (faces et raw)
	@echo "--- Nettoyage des images ---"
	@rm -rf $(FACES_DIR)/*
	@rm -f $(IMG_DIR)/*.jpg
	@echo "Images cleared"

## --- DOCKER ---

check-docker: ## Vérifie que Docker CLI est disponible
	@command -v docker >/dev/null 2>&1 || { \
		echo "Docker CLI introuvable. Installe Docker (ou active le binaire docker) puis réessaie."; \
		exit 1; \
	}

docker-build: check-docker ## Build l'image locale avec le tag du registre
	@echo "Build image tag: $(IMAGE_TAG)"
	docker build -t $(IMAGE_TAG) -f Dockerfile .

docker-smoke: check-docker ## Démarre le container et vérifie que Streamlit répond sur /_stcore/health
	@set -e; \
	container_name="face-detect-smoke"; \
	echo "Suppression éventuelle du container $$container_name..."; \
	docker rm -f $$container_name >/dev/null 2>&1 || true; \
	echo "Démarrage du container $$container_name..."; \
	docker run -d --name $$container_name -p 8501:8501 $(IMAGE_TAG) >/dev/null; \
	echo "Attente du service Streamlit..."; \
	for i in $$(seq 1 20); do \
		if curl -sf http://127.0.0.1:8501/_stcore/health >/dev/null; then \
			echo "Smoke test OK: le container répond."; \
			docker rm -f $$container_name >/dev/null; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "Smoke test KO: le container ne répond pas à temps."; \
	docker logs $$container_name || true; \
	docker rm -f $$container_name >/dev/null 2>&1 || true; \
	exit 1

build-and-push: docker-build docker-smoke ## Build, smoke test puis push sur GitHub
	docker push $(IMAGE_TAG)

release-image: docker-build docker-smoke ## Alias: build + smoke + push
	docker push $(IMAGE_TAG)

release-unstable: docker-build docker-smoke ## Build + smoke + push avec tag versionné (instable)
	docker push $(IMAGE_TAG)

release-stable: docker-build docker-smoke ## Build + smoke + push versionné + latest (stable)
	docker tag $(IMAGE_TAG) $(IMAGE_TAG_LATEST)
	docker push $(IMAGE_TAG)
	docker push $(IMAGE_TAG_LATEST)

show-image-tag: ## Affiche le tag d'image calculé
	@echo "$(IMAGE_TAG)"

deploy-from-registry: ## Lance le projet en utilisant l'image du registre
	IMAGE_NAME=$(IMAGE_TAG) $(DOCKER_COMPOSE) up -d

app-up: ## Lance Docker Compose (build + detach)
	@echo "--- Lancement de Docker Compose ---"
	IMAGE_NAME=$(IMAGE_TAG) $(DOCKER_COMPOSE) -f $(COMPOSE_APP) up -d --build

app-down: ## Arrête et supprime les conteneurs Docker
	$(DOCKER_COMPOSE) -f $(COMPOSE_APP) down

## --- DOCKER DB (Dossier db/) ---

db-up: ## Lance la base de données uniquement
	$(DOCKER_COMPOSE) -f $(COMPOSE_DB) up -d

db-down: ## Arrête la base de données
	$(DOCKER_COMPOSE) -f $(COMPOSE_DB) down

## --- AIDE ---

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
