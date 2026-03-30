from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
)


def create_collection(name="faces"):

    # connections.connect("default", uri=".milvus_demo.db")  # Milvus Lite

    connections.connect(alias="default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="tag_name", dtype=DataType.VARCHAR, max_length=65000),
    ]

    schema = CollectionSchema(fields)
    Collection("faces", schema)


def search(client, name, query_vectors):

    client.search(
        collection_name=name,
        data=[query_vectors[0]],
        limit=2,
        output_fields=["pixels"],
    )

    res = client.query(
        collection_name=name,
        # filter="subject == 'history'",
        output_fields=["pixels"],
    )


create_collection()
