# OpenGovCorpus

A Python library for creating structured datasets and RAG embeddings from government websites.

## Installation

We recommend using [UV](https://github.com/astral-sh/uv) for fast, reliable package installation:

```bash
uv pip install opengovcorpus
```

Or with pip:

```bash
pip install opengovcorpus
```

## Configuration

### Setting Up API Keys

Create a configuration file to store your API credentials:

**Location:** `~/.opengovcorpus/config.json`

```json
{
  "provider": "openai",
  "api_key": "sk-your-api-key-here"
}
```

The library supports multiple embedding providers:

- **OpenAI**: `"provider": "openai"`
- **Gemini**: `"provider": "gemini"`
- **Hugging Face**: `"provider": "huggingface"`

The library automatically reads this configuration file when generating embeddings.

## Usage

### Import the Library

```python
import opengovcorpus as og
```

### 1. Create Dataset

Scrape a government website and create a structured knowledge graph with prompt-response pairs, automatically split into train/validation/test sets.

```python
og.create_dataset(
    name="uk",                    # Name of the dataset folder
    url="https://data.gov.uk",       # Government website to scrape
    include_metadata=True,           # Include metadata for each prompt-response pair
    train_split=0.8,                 # 80% for training
    val_split=0.1,                   # 10% for validation
    test_split=0.1,                  # 10% for testing
    max_pages=10                     # Number of pages to scrape
)
```

**Output Structure:**

```
OpenGovCorpus-uk/
├── train.csv
├── valid.csv
└── test.csv
```

Each CSV contains structured prompt-response pairs suitable for fine-tuning or RAG applications.

### 2. Generate RAG Embeddings

Convert your dataset into vector embeddings for retrieval-augmented generation (RAG) applications.

```python
og.create_rag_embeddings(
    model="openai/text-embedding-3-large",           # Embedding model
    vector_db="chroma",                              # Vector database (Chroma by default)
    config_path="~/.opengovcorpus/config.json"      # Optional: Path to config file (defaults to ~/.opengovcorpus/config.json)
)
```

**Note:** The `config_path` parameter is optional. If not specified, it automatically uses the config file created in the [Configuration](#configuration) section above (`~/.opengovcorpus/config.json`).

**Supported Models:**

- OpenAI: `openai/text-embedding-3-large`, `openai/text-embedding-3-small`
- Gemini: `gemini/text-embedding-004`
- Hugging Face: `hf/sentence-transformers/all-MiniLM-L6-v2`, `hf/BAAI/bge-large-en-v1.5`

**How it works:**

1. Reads `train.csv`, `valid.csv`, and `test.csv` from your dataset
2. Converts prompts (and optionally responses) into vector embeddings
3. Stores embeddings in the specified vector database (Chroma local storage by default)
4. Enables efficient semantic search and retrieval for RAG applications

### Complete Example

```python
import opengovcorpus as og

# Step 1: Create dataset from government website
og.create_dataset(
    name="uk-data",
    url="https://data.gov.uk",
    include_metadata=True,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)

# Step 2: Generate embeddings for RAG
og.create_rag_embeddings(
    model="openai/text-embedding-3-large",
    vector_db="chroma"
    # config_path is optional - uses ~/.opengovcorpus/config.json by default
)
```

## Features

- **Automated Web Scraping**: Extract structured data from government websites
- **Knowledge Graph Creation**: Convert scraped content into meaningful prompt-response pairs
- **Dataset Splitting**: Automatic train/validation/test split configuration
- **Multi-Provider Support**: Works with OpenAI, Gemini, and Hugging Face embeddings
- **Vector Database Integration**: Built-in Chroma support for local embedding storage
- **RAG-Ready**: Outputs are optimized for retrieval-augmented generation workflows

## Use Cases

- Building government data chatbots
- Fine-tuning language models on official documentation
- Creating semantic search systems for public information
- Developing RAG applications for policy and regulation queries
- Generating training datasets for civic tech applications

## Requirements

- Python 3.8+
- API key for your chosen embedding provider (OpenAI, Gemini, or Hugging Face)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

---
