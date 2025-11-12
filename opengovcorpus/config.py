"""
Configuration management for OpenGovCorpus
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from .exceptions import ConfigError


class Config:
    """Configuration manager for API keys and settings"""
    
    DEFAULT_CONFIG_PATH = Path.home() / ".opengovcorpus" / "config.json"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config_data = {}
    
    def load(self) -> Dict:
        """Load configuration from file"""
        if not self.config_path.exists():
            raise ConfigError(
                f"Config file not found at {self.config_path}. "
                f"Please create it with your API credentials."
            )
        
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            return self.config_data
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}")
    
    def get_api_key(self) -> str:
        """Get API key from config"""
        if not self.config_data:
            self.load()
        
        api_key = self.config_data.get("api_key")
        if not api_key:
            raise ConfigError("API key not found in config file")
        
        return api_key
    
    def get_provider(self) -> str:
        """Get provider name from config"""
        if not self.config_data:
            self.load()
        
        provider = self.config_data.get("provider", "openai")
        return provider.lower()
    
    def save(self, provider: str, api_key: str):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "provider": provider,
            "api_key": api_key
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.config_data = config_data


def setup_config(provider: str, api_key: str, config_path: Optional[str] = None):
    """
    Setup configuration file
    
    Args:
        provider: Provider name (openai, gemini, huggingface)
        api_key: API key for the provider
        config_path: Optional custom config path
    """
    config = Config(config_path)
    config.save(provider, api_key)
    print(f"Configuration saved to {config.config_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration
    
    Args:
        config_path: Optional custom config path
        
    Returns:
        Config object
    """
    config = Config(config_path)
    config.load()
    return config