"""
Custom exceptions for OpenGovCorpus
"""


class OpenGovCorpusError(Exception):
    """Base exception for OpenGovCorpus"""
    pass


class ConfigError(OpenGovCorpusError):
    """Raised when there's a configuration error"""
    pass


class ScraperError(OpenGovCorpusError):
    """Raised when scraping fails"""
    pass


class DatasetError(OpenGovCorpusError):
    """Raised when dataset creation fails"""
    pass


class EmbeddingError(OpenGovCorpusError):
    """Raised when embedding generation fails"""
    pass


class VectorStoreError(OpenGovCorpusError):
    """Raised when vector store operations fail"""
    pass