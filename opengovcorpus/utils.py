"""
Utility functions for OpenGovCorpus
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()


def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid
    
    Args:
        url: URL to check
        
    Returns:
        True if valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def get_domain(url: str) -> str:
    """
    Extract domain from URL
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    parsed = urlparse(url)
    return parsed.netloc


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def save_to_csv(data: List[Dict[str, Any]], filepath: str):
    """
    Save data to CSV file
    
    Args:
        data: List of dictionaries
        filepath: Path to save CSV
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(data)} records to {filepath}")


def load_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dictionaries
    """
    import pandas as pd
    from pathlib import Path
    
    # Check if file is empty
    file_path = Path(filepath)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []
    
    try:
        df = pd.read_csv(filepath)
        # Check if dataframe is empty
        if df.empty:
            return []
        return df.to_dict('records')
    except pd.errors.EmptyDataError:
        return []