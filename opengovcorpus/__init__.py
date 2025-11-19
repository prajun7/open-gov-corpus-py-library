"""
OpenGovCorpus - A library for creating datasets and RAG embeddings from government websites
"""

__version__ = "0.1.4"
__author__ = "Prajun Trital"
__license__ = "MIT"

# Import main functions that users will call
from .dataset import create_dataset
from .embeddings import create_rag_embeddings
from .config import setup_config, load_config

# Define what gets imported with "from opengovcorpus import *"
__all__ = [
    "create_dataset",
    "create_rag_embeddings",
    "setup_config",
    "load_config",
]

