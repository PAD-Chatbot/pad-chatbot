"""
config.py  –  single source of truth for constants
Author: Brandon Desbiens

Holds:
- File-system paths (PDF, index, chunk pickle)
- Ollama model / endpoint details
- Embedding model used for retrieval
- Tunables for retrieval (top-k) and generation (temperature)

Any time you need to change the model, move the files, or fiddle with
hyperparameters, do it here and nowhere else. All other modules import
`config` instead of hard-coding values.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# Paths (file locations)
PDF_PATH = BASE_DIR / "pad.pdf"  # We need to add the French PAD later and rerun ingest.py
INDEX_FILE = BASE_DIR / "pad.index"
CHUNK_FILE = BASE_DIR / "pad_chunks.pkl"

# Model & retrieval parameters
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b-instruct"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4 # Number of chunks to stuff into prompt
LLM_TEMPERATURE = 0.2
