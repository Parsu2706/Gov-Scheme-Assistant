import json

from fastembed import SparseTextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (CHUNKS_PATH, COLLECTION_NAME, EMBEDDING_DIM,
                    EMBEDDING_MODEL, SPARSE_MODEL)
from ingestion.store import get_client


def load_chunks(): 
    with open(CHUNKS_PATH , encoding="utf-8") as f: 
        return [json.loads(row) for row in f]


def build_index(chunks): 
    client = get_client()
    if client.collection_exists(COLLECTION_NAME): 
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME , 
        vectors_config = {"dense": models.VectorParams(size = EMBEDDING_DIM , distance = models.Distance.COSINE)} , 
        sparse_vectors_config = {"bm25" : models.SparseVectorParams(modifier = models.Modifier.IDF)}
    )
    dense_model = SentenceTransformer(EMBEDDING_MODEL)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    batch = 256 
    for start in range(0 , len(chunks) , batch): 
        parts = chunks[start:start + batch]
        texts = [c["text"] for c in parts]
        dense = dense_model.encode(texts , normalize_embeddings = True)
        sparse = list(sparse_model.embed(texts))

        points = []
        for i , chunk in enumerate(parts): 
            points.append(models.PointStruct(
                id = start + i , 
                vector = {
                    "dense" : dense[i].tolist() , 
                    "bm25" : models.SparseVector(indices = sparse[i].indices.tolist() , 
                                                 values = sparse[i].values.tolist())
                }, 
                payload = {"text" : chunk["text"] , **chunk["metadata"]}
            ))
        client.upsert(COLLECTION_NAME , points)

if __name__ == "__main__":
    chunks = load_chunks()
    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks")