"""
Tests for scraper module
"""

import pytest
from opengovcorpus.scraper import GovernmentScraper
from opengovcorpus.exceptions import ScraperError


def test_invalid_url():
    """Test that invalid URL raises error"""
    with pytest.raises(ScraperError):
        GovernmentScraper("not-a-url")


def test_scraper_initialization():
    """Test scraper initializes correctly"""
    scraper = GovernmentScraper("https://data.gov")
    assert scraper.base_url == "https://data.gov"
    assert scraper.domain == "data.gov"


# Note: Add more tests with mocked HTTP responses