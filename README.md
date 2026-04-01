# Face Detect Laforge

Application Streamlit d’annotation et de recherche de visages avec stockage vectoriel Milvus.

## Structure modulaire

```text
app/
  vision/        # Détection et extraction des visages (OpenCV)
  ml/            # Génération d'embeddings (ResNet)
  storage/       # Accès Milvus (collection, insert, KNN, index)
  services/      # Orchestration métier (pipeline upload → embeddings → suggestions)
  workers/       # Exécution asynchrone locale (queue + thread)
  ui/            # Interface Streamlit
scripts/
  rebuild_index.py  # Rebuild batch des embeddings/index
view/
  display.py     # Point d’entrée Streamlit (compat)
```

## Lancer localement

1. Installer et préparer:

```bash
make all
```

2. Lancer toute la stack (app + Milvus):

```bash
make app-up
```

Cette commande démarre automatiquement `etcd`, `minio`, `standalone` (Milvus) puis l'application Streamlit avec `MILVUS_HOST=standalone`.

## Fonctionnement

### Vectorisation

- Upload d’image dans l’UI.
- Un worker local en arrière-plan traite l’image:
  1) détection des visages,
  2) génération des embeddings ResNet,
  3) recherche de suggestions KNN dans Milvus.
- À validation d’annotation, chaque embedding est inséré en base Milvus.
- Si aucun label n’est saisi, le visage est stocké en `__unknown__`.

### Recherche par similarité

- Requête KNN via `Collection.search` Milvus.
- Index IVF_FLAT + métrique COSINE.
- Résultats retournés avec nom, distance et flag `is_unknown`.
- L'UI affiche le score de similarité au format pourcentage.

### Annotation

- Pendant la première annotation, chaque visage détecté peut être exclu.
- Les visages exclus ne sont ni sauvegardés dans Milvus, ni affichés dans la galerie/historique.
- La galerie est affichée en grille pour faciliter la revue rapide.

### Mise à jour de l’index

- Mise à jour automatique après accumulation de `REBUILD_THRESHOLD` nouveaux visages.
- Script batch manuel disponible:

```bash
uv run python scripts/rebuild_index.py --folder data/images/faces --label Alice
```

### Traitement en arrière-plan

- Implémenté par `BackgroundTaskRunner` (queue locale + thread daemon).
- L’UI Streamlit ne bloque pas pendant détection/vectorisation.

## Tests ciblés

```bash
python -m unittest discover -s tests -v
```

## Docker (build, smoke test, push)

Le projet expose des commandes Make dédiées :

```bash
# Build de l'image taggée pour GHCR
make docker-build

# Vérifie que le container démarre et répond sur l'endpoint health Streamlit
make docker-smoke

# Push dans le registre configuré (IMAGE_TAG dans le Makefile)
make build-and-push
```
