"""
Basic usage example for OpenGovCorpus
"""

import opengovcorpus as og

# Example 1: Setup configuration (run once)
og.setup_config(
    provider="openai",
    api_key="your-api-key-here"
)

# Example 2: Create a dataset
print("Creating dataset...")
og.create_dataset(
    name="usa-data",
    url="https://www.gov.uk/"
    include_metadata=True,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    max_pages=10  # Limit for testing
)

# Example 3: Generate embeddings
print("\nGenerating embeddings...")
og.create_rag_embeddings(
    model="openai/text-embedding-3-large",
    vector_db="chroma"
)

print("\nDone! Check the OpenGovCorpus-usa-data directory for results.")