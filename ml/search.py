"""Legacy wrapper for Milvus search."""

from app.storage.milvus_store import MilvusFaceStore


def search_embedding(query_embedding, collection=None):
    if collection is not None:
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=3,
        )
        return results
    return MilvusFaceStore().search(query_embedding)
