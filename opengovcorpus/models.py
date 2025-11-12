"""
Data models for OpenGovCorpus
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class PromptResponse:
    """Represents a prompt-response pair"""
    prompt: str
    response: str
    metadata: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
            "url": self.url,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class ScrapedContent:
    """Represents scraped content from a URL"""
    url: str
    title: str
    content: str
    links: list
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "links": self.links,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset creation"""
    name: str
    url: str
    include_metadata: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_pages: Optional[int] = None
    
    def __post_init__(self):
        """Validate splits"""
        total = self.train_split + self.val_split + self.test_split
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Splits must sum to 1.0, got {total}")