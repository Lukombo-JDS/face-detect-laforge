from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config import SETTINGS
from app.logging import get_logger

LOGGER = get_logger(__name__)

MILVUS_METRIC_TYPE = "IP"
MILVUS_HNSW_INDEX_PARAMS = {
    "metric_type": MILVUS_METRIC_TYPE,
    "index_type": "HNSW",
    "params": {"M": 32, "efConstruction": 200},
}
DEFAULT_SEARCH_EF = 128


def build_search_params(ef: int = DEFAULT_SEARCH_EF, metric_type: str = MILVUS_METRIC_TYPE) -> dict[str, object]:
    """Centralise les paramètres de recherche Milvus pour ajuster facilement le recall/latence."""
    return {"metric_type": metric_type, "params": {"ef": ef}}


@dataclass(frozen=True)
class SearchResult:
    person_name: str
    distance: float
    is_unknown: bool
    similarity_percent: float


class MilvusFaceStore:
    def __init__(self, collection_name: str = SETTINGS.milvus_collection) -> None:
        self.collection_name = collection_name
        self._connected = False
        self._collection: Collection | None = None

    def connect(self) -> None:
        if self._connected:
            return
        connections.connect(
            host=SETTINGS.milvus_host,
            port=SETTINGS.milvus_port,
            timeout=1,
        )
        self._connected = True
        self._collection = self._ensure_collection()

    def _ensure_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="is_unknown", dtype=DataType.BOOL),
            FieldSchema(name="source_image", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=SETTINGS.embedding_dim),
        ]
        schema = CollectionSchema(fields=fields, description="Face embeddings")
        collection = Collection(name=self.collection_name, schema=schema)
        self.ensure_index(collection)
        return collection

    def ensure_index(self, collection: Collection | None = None) -> None:
        target = collection or self.collection
        if target.indexes:
            return
        target.create_index(
            field_name="embedding",
            index_params=MILVUS_HNSW_INDEX_PARAMS,
        )
        LOGGER.info("milvus_index_ready collection=%s", target.name)

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Milvus non initialisé. Appelez connect() avant usage")
        return self._collection

    def add_face(self, person_name: str, embedding: np.ndarray, source_image: str) -> None:
        normalized_name = person_name.strip() or SETTINGS.unknown_label
        is_unknown = normalized_name == SETTINGS.unknown_label
        normalized_embedding = self._normalize_embedding(embedding)
        self.collection.insert(
            [
                [normalized_name],
                [is_unknown],
                [source_image],
                [normalized_embedding.astype(np.float32).tolist()],
            ]
        )

    def search(self, query: np.ndarray, limit: int = 3, ef: int = DEFAULT_SEARCH_EF) -> list[SearchResult]:
        self.ensure_index()
        if not self.collection.indexes:
            raise RuntimeError("Aucun index Milvus disponible: impossible de charger la collection")
        self.collection.load()
        normalized_query = self._normalize_embedding(query)
        metric_type = self._resolve_metric_type(self.collection)
        results = self.collection.search(
            data=[normalized_query.astype(np.float32).tolist()],
            anns_field="embedding",
            param=build_search_params(ef=ef, metric_type=metric_type),
            limit=limit,
            output_fields=["person_name", "is_unknown"],
        )
        parsed: list[SearchResult] = []
        for hit in results[0]:
            parsed.append(
                SearchResult(
                    person_name=hit.entity.get("person_name"),
                    distance=float(hit.distance),
                    is_unknown=bool(hit.entity.get("is_unknown")),
                    similarity_percent=self._score_to_percent(float(hit.distance)),
                )
            )
        return parsed

    @staticmethod
    def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
        casted = vector.astype(np.float32)
        norm = float(np.linalg.norm(casted))
        if norm == 0.0:
            return casted
        return casted / norm

    @staticmethod
    def _resolve_metric_type(collection: Collection) -> str:
        """Garde la compatibilité avec les collections existantes (L2) au lieu de casser la recherche."""
        if not collection.indexes:
            return MILVUS_METRIC_TYPE

        first_index = collection.indexes[0]
        index_params = None
        if hasattr(first_index, "to_dict"):
            try:
                index_info = first_index.to_dict()
            except Exception:  # pragma: no cover - dépend du client Milvus utilisé
                index_info = {}
            index_params = index_info.get("index_param", index_info)
        elif hasattr(first_index, "params"):
            index_params = getattr(first_index, "params")

        if isinstance(index_params, dict):
            metric_type = index_params.get("metric_type")
            if isinstance(metric_type, str) and metric_type:
                if metric_type != MILVUS_METRIC_TYPE:
                    LOGGER.warning(
                        "milvus_metric_compatibility collection=%s metric=%s",
                        collection.name,
                        metric_type,
                    )
                return metric_type
        return MILVUS_METRIC_TYPE

    @staticmethod
    def _score_to_percent(score: float) -> float:
        return max(0.0, min(score, 1.0)) * 100.0
