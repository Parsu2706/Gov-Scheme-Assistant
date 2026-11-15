from fastembed import SparseTextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL, QUERY_PREFIX, SPARSE_MODEL
from ingestion.store import get_client


class Retriever:
    def __init__(self):
        self.client = get_client()
        self.dense_model = SentenceTransformer(EMBEDDING_MODEL , device = "cpu")
        self.sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    def search(self, query, k=5, state=None, mode="hybrid" , scheme = None):
        dense = self.dense_model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        sp = list(self.sparse_model.query_embed(query))[0]
        sparse = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())

        conditions = []
        if state: 
            conditions.append(models.FieldCondition(key="state" , match = models.MatchValue(value = state)))

        if scheme:
            conditions.append(models.FieldCondition(
                key="scheme_name",
                match=models.MatchValue(value=scheme)
            ))

        flt = models.Filter(must=conditions) if conditions else None

        if mode == "dense":
            result = self.client.query_points(
                COLLECTION_NAME, query=dense, using="dense",
                limit=k, query_filter=flt, with_payload=True)
        elif mode == "bm25":
            result = self.client.query_points(
                COLLECTION_NAME, query=sparse, using="bm25",
                limit=k, query_filter=flt, with_payload=True)
        else:
            result = self.client.query_points(
                COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=30, filter=flt),
                    models.Prefetch(query=sparse, using="bm25", limit=30, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k, with_payload=True)

        return [p.payload for p in result.points]


if __name__ == "__main__":
    r = Retriever()
    for hit in r.search("scholarship for girl students"):
        print(hit["scheme_name"], "|", hit["section"])