"""Shared defaults and readable names for the five retrieval methods."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'scifact'
ARTIFACT_DIR = ROOT / 'artifacts'
MODEL_DIR = ROOT / 'models'
DEFAULT_MODEL = MODEL_DIR / 'embeddinggemma-300m-Q4_0.gguf'
METHODS = {
    'baseline': 'Embedding only',
    'mse': 'Embedding + AE: MSE',
    'mse_cosine': 'Embedding + AE: MSE + Cosine',
    'mse_cosine_contractive': 'Embedding + AE: MSE + Cosine + Contractive',
    'mse_contractive': 'Embedding + AE: MSE + Contractive',
}
AE_METHODS = tuple(k for k in METHODS if k != 'baseline')
DIMENSION = 768
CONTEXT = 2048
PROMPT_VERSION = 'embeddinggemma-retrieval-v1'
