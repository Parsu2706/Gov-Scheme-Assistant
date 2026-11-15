from qdrant_client import QdrantClient
from config import QDRANT_PATH

def get_client(): 
    return QdrantClient(path = str(QDRANT_PATH))
