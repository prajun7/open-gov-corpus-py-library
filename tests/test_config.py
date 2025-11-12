"""
Tests for config module
"""

import pytest
import json
from pathlib import Path
import tempfile

from opengovcorpus.config import Config, setup_config, load_config
from opengovcorpus.exceptions import ConfigError


def test_config_save_and_load():
    """Test saving and loading config"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        
        # Save config
        config = Config(str(config_path))
        config.save("openai", "test-key-123")
        
        # Load config
        loaded_config = Config(str(config_path))
        loaded_config.load()
        
        assert loaded_config.get_provider() == "openai"
        assert loaded_config.get_api_key() == "test-key-123"


def test_config_missing_file():
    """Test error when config file missing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "missing.json"
        config = Config(str(config_path))
        
        with pytest.raises(ConfigError):
            config.load()


def test_setup_config():
    """Test setup_config function"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        
        setup_config("gemini", "gemini-key", str(config_path))
        
        # Verify file was created
        assert config_path.exists()
        
        # Verify contents
        with open(config_path) as f:
            data = json.load(f)
        
        assert data["provider"] == "gemini"
        assert data["api_key"] == "gemini-key"