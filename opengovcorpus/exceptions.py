"""
Custom exceptions for OpenGovCorpus
exceptions.py does not raise exceptions itself. 
It defines exception classes (types) that other modules use.

Exception Hierarchy:
- OpenGovCorpusError (base)
  - ConfigError: Configuration and API key issues
  - ValidationError: Input validation failures
  - ScraperError: Web scraping failures
    - NetworkError: Network/HTTP connection issues
  - DatasetError: Dataset creation and file operations
    - FileError: File I/O operations
  - EmbeddingError: Embedding generation failures
    - ProviderError: Provider-specific API/authentication issues
  - VectorStoreError: Vector database operations
"""


class OpenGovCorpusError(Exception):
    """Base exception for OpenGovCorpus"""
    pass


class ConfigError(OpenGovCorpusError):
    """
    Raised when there's a configuration error.
    
    Examples:
    - Config file not found
    - Invalid JSON in config file
    - API key missing or invalid
    - Invalid provider name
    """
    pass


class ValidationError(OpenGovCorpusError):
    """
    Raised when input validation fails.
    
    Examples:
    - Invalid split ratios (must sum to 1.0)
    - Invalid model format (must be 'provider/model-name')
    - Invalid URL format
    - Invalid parameter values
    """
    pass


class ScraperError(OpenGovCorpusError):
    """
    Raised when web scraping fails.
    
    Examples:
    - Invalid URL provided
    - Failed to scrape a page
    - Parsing errors
    """
    pass


class NetworkError(ScraperError):
    """
    Raised when network/HTTP operations fail.
    
    Examples:
    - Connection timeout
    - HTTP error status codes
    - Network unreachable
    - SSL/TLS errors
    """
    pass


class DatasetError(OpenGovCorpusError):
    """
    Raised when dataset creation fails.
    
    Examples:
    - No content scraped from website
    - No prompt-response pairs generated
    - Dataset directory creation failed
    """
    pass


class FileError(DatasetError):
    """
    Raised when file I/O operations fail.
    
    Examples:
    - CSV file read/write errors
    - File not found
    - Permission denied
    - Disk full
    - Invalid file format
    """
    pass


class EmbeddingError(OpenGovCorpusError):
    """
    Raised when embedding generation fails.
    
    Examples:
    - Embedding API call failed
    - Invalid model name
    - Batch processing errors
    - No dataset found
    """
    pass


class ProviderError(EmbeddingError):
    """
    Raised when provider-specific operations fail.
    
    Examples:
    - API authentication failure
    - Provider package not installed
    - Rate limiting exceeded
    - Invalid API key for provider
    - Provider service unavailable
    """
    pass


class VectorStoreError(OpenGovCorpusError):
    """
    Raised when vector store operations fail.
    
    Examples:
    - Vector database connection failed
    - Failed to add embeddings
    - Query operation failed
    - Unknown vector store type
    - ChromaDB package not installed
    """
    pass