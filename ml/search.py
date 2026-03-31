# TODO: Faire la recherche vectoriel


def search_embedding(query_embedding, collection):
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
    )
    return results
