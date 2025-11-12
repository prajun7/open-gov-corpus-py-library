"""
Dataset creation and management
"""

import random
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .models import DatasetConfig, PromptResponse, ScrapedContent
from .scraper import scrape_website
from .exceptions import DatasetError
from .utils import ensure_directory, chunk_text


class DatasetCreator:
    """Creates datasets from scraped content"""
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset creator
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.output_dir = ensure_directory(f"OpenGovCorpus-{config.name}")
    
    def create(self) -> dict:
        """
        Create the dataset
        
        Returns:
            Dictionary with file paths
        """
        print(f"Starting dataset creation for: {self.config.name}")
        print(f"Scraping URL: {self.config.url}")
        
        # Step 1: Scrape website
        scraped_content = scrape_website(
            self.config.url,
            max_pages=self.config.max_pages
        )
        
        if not scraped_content:
            raise DatasetError("No content scraped from website")
        
        print(f"Scraped {len(scraped_content)} pages")
        
        # Step 2: Convert to prompt-response pairs
        prompt_responses = self._create_prompt_responses(scraped_content)
        
        if not prompt_responses:
            raise DatasetError("No prompt-response pairs generated")
        
        print(f"Generated {len(prompt_responses)} prompt-response pairs")
        
        # Step 3: Split into train/val/test
        splits = self._split_data(prompt_responses)
        
        # Step 4: Save to CSV files
        file_paths = self._save_splits(splits)
        
        print(f"Dataset created successfully in: {self.output_dir}")
        return file_paths
    
    def _create_prompt_responses(self, content: List[ScrapedContent]) -> List[PromptResponse]:
        """
        Convert scraped content to prompt-response pairs
        
        Args:
            content: List of scraped content
            
        Returns:
            List of prompt-response pairs
        """
        pairs = []
        
        for item in content:
            # Chunk long content
            chunks = chunk_text(item.content, chunk_size=1000, overlap=100)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 100:  # Skip very short chunks
                    continue
                
                # Create prompt from title and chunk number
                prompt = f"What information is available about '{item.title}'?"
                if len(chunks) > 1:
                    prompt += f" (Part {i+1}/{len(chunks)})"
                
                # Metadata
                metadata = None
                if self.config.include_metadata:
                    metadata = {
                        'source_url': item.url,
                        'title': item.title,
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'timestamp': item.timestamp.isoformat()
                    }
                
                pairs.append(PromptResponse(
                    prompt=prompt,
                    response=chunk,
                    metadata=metadata,
                    url=item.url,
                    timestamp=item.timestamp
                ))
        
        return pairs
    
    def _split_data(self, data: List[PromptResponse]) -> dict:
        """
        Split data into train/val/test
        
        Args:
            data: List of prompt-response pairs
            
        Returns:
            Dictionary with splits
        """
        # Shuffle data
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * self.config.train_split)
        val_end = train_end + int(total * self.config.val_split)
        
        return {
            'train': data[:train_end],
            'valid': data[train_end:val_end],
            'test': data[val_end:]
        }
    
    def _save_splits(self, splits: dict) -> dict:
        """
        Save splits to CSV files
        
        Args:
            splits: Dictionary with train/val/test splits
            
        Returns:
            Dictionary with file paths
        """
        file_paths = {}
        
        for split_name, split_data in splits.items():
            filepath = self.output_dir / f"{split_name}.csv"
            
            # Convert to dictionaries
            rows = [item.to_dict() for item in split_data]
            
            # Save to CSV
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            
            file_paths[split_name] = str(filepath)
            print(f"  {split_name}: {len(split_data)} samples -> {filepath}")
        
        return file_paths


def create_dataset(
    name: str,
    url: str,
    include_metadata: bool = True,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_pages: Optional[int] = None
) -> dict:
    """
    Create a dataset from a government website
    
    Args:
        name: Name of the dataset
        url: Government website URL
        include_metadata: Include metadata in output
        train_split: Training set fraction
        val_split: Validation set fraction
        test_split: Test set fraction
        max_pages: Maximum pages to scrape
        
    Returns:
        Dictionary with file paths
    """
    config = DatasetConfig(
        name=name,
        url=url,
        include_metadata=include_metadata,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        max_pages=max_pages
    )
    
    creator = DatasetCreator(config)
    return creator.create()