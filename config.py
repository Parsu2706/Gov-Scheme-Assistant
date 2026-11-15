from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "final_gov_myscheme_data.csv"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
QDRANT_PATH = BASE_DIR / "qdrant_data"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384              # output size of bge-small-en-v1.5
SPARSE_MODEL = "Qdrant/bm25"     # keyword (BM25) model from fastembed

# Qdrant
COLLECTION_NAME = "schemes"

# CSV column -> section label used in chunks
SECTIONS = {
    "Description": "Description",
    "Eligibility Criteria": "Eligibility",
    "Eligibility (General)": "General Eligibility",
    "Exclusions / Ineligibility": "Exclusions",
    "Benefits": "Benefits",
    "Application Process": "Application Process",
    "Documents Required": "Documents Required",
    "Frequently Asked Questions (FAQs)": "FAQ",
}

# Cleaning rules
MINISTRY_FIXES = {
    "Ministry Of C": "Ministry of Communications",
    "Ministry Of E": "Ministry of Environment, Forest and Climate Change",
    "Ministry Of R": "Ministry of Road Transport and Highways",
    "Ministry Of Y": "Ministry of Youth Affairs and Sports",
}

HEALTH_KEYWORDS = ["health", "medical", "icmr", "fssai", "food safety", "disease"]
# LLM (Groq)
LLM_MODEL = "openai/gpt-oss-120b"
REWRITE_MODEL = LLM_MODEL
NOT_FOUND = "I couldn't find this in the available government schemes."