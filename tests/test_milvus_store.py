from unittest.mock import MagicMock

import numpy as np

from app.storage.milvus_store import (
    DEFAULT_SEARCH_EF,
    MILVUS_HNSW_INDEX_PARAMS,
    build_search_params,
    MilvusFaceStore,
)


def test_build_search_params_defaults_to_ef_128() -> None:
    assert build_search_params() == {"metric_type": "IP", "params": {"ef": DEFAULT_SEARCH_EF}}


def test_build_search_params_accepts_override_metric() -> None:
    assert build_search_params(metric_type="L2") == {"metric_type": "L2", "params": {"ef": DEFAULT_SEARCH_EF}}


def test_ensure_index_creates_hnsw_index_when_missing() -> None:
    store = MilvusFaceStore(collection_name_tagged="test_collection")
    collection = MagicMock()
    collection.indexes = []
    collection.name = "test_collection"

    store.ensure_index(collection)

    collection.create_index.assert_called_once_with(
        field_name="embedding",
        index_params=MILVUS_HNSW_INDEX_PARAMS,
    )


def test_search_ensures_index_then_loads_and_uses_ef() -> None:
    store = MilvusFaceStore(collection_name_tagged="test_collection")
    collection = MagicMock()
    collection.indexes = [object()]

    hit = MagicMock()
    hit.id = 7
    hit.entity.get.side_effect = lambda key: {
        "person_name": "alice",
        "annotation": "alice",
        "is_unknown": False,
    }[key]
    hit.distance = 0.91
    collection.search.return_value = [[hit]]

    store._tagged_collection = collection
    store._connected = True

    results = store.search(np.array([1.0, 2.0, 3.0], dtype=np.float32), limit=3, ef=256)

    collection.load.assert_called_once()
    collection.search.assert_called_once()
    call_kwargs = collection.search.call_args.kwargs
    assert call_kwargs["param"] == {"metric_type": "IP", "params": {"ef": 256}}
    assert results[0].person_name == "alice"
    assert results[0].annotation == "alice"
    assert results[0].face_id == 7


def test_search_uses_existing_collection_metric_type() -> None:
    store = MilvusFaceStore(collection_name_tagged="test_collection")
    collection = MagicMock()
    index = MagicMock()
    index.to_dict.return_value = {"index_param": {"metric_type": "L2"}}
    collection.indexes = [index]
    collection.name = "test_collection"

    hit = MagicMock()
    hit.id = 8
    hit.entity.get.side_effect = lambda key: {
        "person_name": "bob",
        "annotation": "bob",
        "is_unknown": False,
    }[key]
    hit.distance = 0.42
    collection.search.return_value = [[hit]]

    store._tagged_collection = collection
    store._connected = True

    store.search(np.array([1.0, 2.0, 3.0], dtype=np.float32), limit=2, ef=64)

    call_kwargs = collection.search.call_args.kwargs
    assert call_kwargs["param"] == {"metric_type": "L2", "params": {"ef": 64}}


def test_score_to_percent_l2_converts_short_distance_to_high_similarity() -> None:
    high = MilvusFaceStore._score_to_percent(0.1, "L2")
    low = MilvusFaceStore._score_to_percent(2.0, "L2")
    assert high > low


def test_add_face_routes_unknown_and_tagged_collections() -> None:
    store = MilvusFaceStore()
    tagged = MagicMock()
    unknown = MagicMock()
    store._tagged_collection = tagged
    store._unknown_collection = unknown
    store._connected = True

    store.add_face("Alice", np.array([1.0, 0.0], dtype=np.float32), source_image="img1.jpg")
    store.add_face("   ", np.array([1.0, 0.0], dtype=np.float32), source_image="img2.jpg")

    assert tagged.insert.call_count == 1
    assert unknown.insert.call_count == 1
