"""
Basic usage example for OpenGovCorpus
"""

import opengovcorpus as og
from pathlib import Path

# Example 1: Setup configuration (run once)
# If you have a config.json in this folder, skip this step
# Otherwise, uncomment and update with your API key:
# og.setup_config(
#     provider="gemini",  # or "openai" or "huggingface"
#     api_key="your-api-key-here"
# )

# Example 2: Create a dataset
print("Creating dataset...")
og.create_dataset(
    name="gov-uk-data",
    url="https://www.gov.uk/",
    include_metadata=True,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    max_pages=10  # Limit for testing
)

# Example 3: Generate embeddings
# Use the local config.json file in this folder
config_path = Path(__file__).parent / "config.json"
print("\nGenerating embeddings...")
og.create_rag_embeddings(
    model="gemini/text-embedding-004",  # Change to "openai/text-embedding-3-large" for OpenAI
    vector_db="chroma",
    config_path=str(config_path)  # Use local config file
)

print("\nDone! Check the OpenGovCorpus-uk-data directory for results.")