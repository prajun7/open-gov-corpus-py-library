"""
Tests for scraper module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from opengovcorpus.scraper import GovernmentScraper, scrape_website
from opengovcorpus.exceptions import ScraperError, NetworkError
from opengovcorpus.models import ScrapedContent


def test_invalid_url():
    """Test that invalid URL raises error"""
    with pytest.raises(ScraperError, match="Invalid URL"):
        GovernmentScraper("not-a-url")
    
    with pytest.raises(ScraperError):
        GovernmentScraper("")
    
    with pytest.raises(ScraperError):
        GovernmentScraper("http://")


def test_scraper_initialization():
    """Test scraper initializes correctly"""
    scraper = GovernmentScraper("https://data.gov.uk")
    assert scraper.base_url == "https://data.gov.uk"
    assert scraper.domain == "data.gov.uk"
    assert scraper.max_pages is None
    assert scraper.visited_urls == set()
    
    # Test with max_pages
    scraper = GovernmentScraper("https://data.gov.uk", max_pages=5)
    assert scraper.max_pages == 5


def test_scraper_user_agent():
    """Test that scraper sets correct User-Agent"""
    scraper = GovernmentScraper("https://example.com")
    assert "OpenGovCorpus" in scraper.session.headers["User-Agent"]


@patch('opengovcorpus.scraper.requests.Session')
def test_scrape_page_success(mock_session):
    """Test successful page scraping"""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"""
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Content</h1>
            <p>This is test content for scraping.</p>
            <a href="/page2">Link 1</a>
            <a href="https://example.com/page3">Link 2</a>
        </body>
    </html>
    """
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raise_for_status = Mock()
    
    # Setup session mock
    mock_session_instance = Mock()
    mock_session_instance.get.return_value = mock_response
    mock_session_instance.headers = {}
    mock_session.return_value = mock_session_instance
    
    scraper = GovernmentScraper("https://example.com")
    scraper.session = mock_session_instance
    
    content = scraper._scrape_page("https://example.com")
    
    assert content is not None
    assert isinstance(content, ScrapedContent)
    assert content.url == "https://example.com"
    assert "Test Page" in content.title
    assert "test content" in content.content.lower()
    assert len(content.links) > 0
    assert content.metadata["status_code"] == 200
    assert isinstance(content.timestamp, datetime)


@patch('opengovcorpus.scraper.requests.Session')
def test_scrape_page_http_error(mock_session):
    """Test scraping page with HTTP error"""
    # Mock response with 404 error
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    
    mock_session_instance = Mock()
    mock_session_instance.get.return_value = mock_response
    mock_session_instance.headers = {}
    mock_session.return_value = mock_session_instance
    
    scraper = GovernmentScraper("https://example.com")
    scraper.session = mock_session_instance
    
    with pytest.raises(ScraperError, match="Failed to scrape"):
        scraper._scrape_page("https://example.com/notfound")


@patch('opengovcorpus.scraper.requests.Session')
def test_scrape_page_timeout(mock_session):
    """Test scraping page with timeout"""
    mock_session_instance = Mock()
    mock_session_instance.get.side_effect = Exception("Connection timeout")
    mock_session_instance.headers = {}
    mock_session.return_value = mock_session_instance
    
    scraper = GovernmentScraper("https://example.com")
    scraper.session = mock_session_instance
    
    with pytest.raises(ScraperError, match="Failed to scrape"):
        scraper._scrape_page("https://example.com")


@patch('opengovcorpus.scraper.GovernmentScraper._scrape_page')
@patch('opengovcorpus.scraper.time.sleep')  # Mock sleep to speed up tests
def test_scrape_with_max_pages(mock_sleep, mock_scrape_page):
    """Test scraping with max_pages limit"""
    # Create mock scraped content
    mock_content = ScrapedContent(
        url="https://example.com",
        title="Test Page",
        content="Test content",
        links=["https://example.com/page2"],
        metadata={"status_code": 200},
        timestamp=datetime.now()
    )
    
    mock_scrape_page.return_value = mock_content
    
    scraper = GovernmentScraper("https://example.com", max_pages=2)
    results = scraper.scrape()
    
    assert len(results) == 1  # Only base URL scraped
    assert mock_scrape_page.call_count == 1


@patch('opengovcorpus.scraper.GovernmentScraper._scrape_page')
@patch('opengovcorpus.scraper.time.sleep')
def test_scrape_visits_links(mock_sleep, mock_scrape_page):
    """Test that scraper follows links within same domain"""
    # First page with links
    page1 = ScrapedContent(
        url="https://example.com",
        title="Page 1",
        content="Content 1",
        links=["https://example.com/page2", "https://other.com/page"],
        metadata={"status_code": 200},
        timestamp=datetime.now()
    )
    
    # Second page
    page2 = ScrapedContent(
        url="https://example.com/page2",
        title="Page 2",
        content="Content 2",
        links=[],
        metadata={"status_code": 200},
        timestamp=datetime.now()
    )
    
    # Mock to return different content for different URLs
    def mock_scrape(url):
        if url == "https://example.com":
            return page1
        elif url == "https://example.com/page2":
            return page2
        return None
    
    mock_scrape_page.side_effect = mock_scrape
    
    scraper = GovernmentScraper("https://example.com", max_pages=5)
    results = scraper.scrape()
    
    # Should scrape both pages (but not other.com)
    assert len(results) == 2
    urls_scraped = [r.url for r in results]
    assert "https://example.com" in urls_scraped
    assert "https://example.com/page2" in urls_scraped
    assert "https://other.com/page" not in urls_scraped


@patch('opengovcorpus.scraper.GovernmentScraper._scrape_page')
@patch('opengovcorpus.scraper.time.sleep')
def test_scrape_skips_visited_urls(mock_sleep, mock_scrape_page):
    """Test that scraper skips already visited URLs"""
    mock_content = ScrapedContent(
        url="https://example.com",
        title="Test",
        content="Content",
        links=["https://example.com"],  # Link back to itself
        metadata={"status_code": 200},
        timestamp=datetime.now()
    )
    
    mock_scrape_page.return_value = mock_content
    
    scraper = GovernmentScraper("https://example.com", max_pages=5)
    results = scraper.scrape()
    
    # Should only scrape once, not loop infinitely
    assert len(results) == 1
    assert len(scraper.visited_urls) == 1


@patch('opengovcorpus.scraper.GovernmentScraper._scrape_page')
@patch('opengovcorpus.scraper.time.sleep')
def test_scrape_handles_errors_gracefully(mock_sleep, mock_scrape_page):
    """Test that scraper continues on errors"""
    # First call succeeds, second fails
    mock_content = ScrapedContent(
        url="https://example.com",
        title="Test",
        content="Content",
        links=["https://example.com/page2"],
        metadata={"status_code": 200},
        timestamp=datetime.now()
    )
    
    def mock_scrape(url):
        if url == "https://example.com":
            return mock_content
        else:
            raise ScraperError("Failed")
    
    mock_scrape_page.side_effect = mock_scrape
    
    scraper = GovernmentScraper("https://example.com", max_pages=5)
    results = scraper.scrape()
    
    # Should return successful scrapes even if some fail
    assert len(results) == 1


def test_scrape_website_function():
    """Test the scrape_website convenience function"""
    with patch('opengovcorpus.scraper.GovernmentScraper') as mock_scraper_class:
        mock_scraper = Mock()
        mock_scraper.scrape.return_value = []
        mock_scraper_class.return_value = mock_scraper
        
        result = scrape_website("https://example.com", max_pages=3)
        
        mock_scraper_class.assert_called_once_with("https://example.com", max_pages=3)
        mock_scraper.scrape.assert_called_once()
        assert result == []


@patch('opengovcorpus.scraper.GovernmentScraper._scrape_page')
@patch('opengovcorpus.scraper.time.sleep')
def test_scraped_content_structure(mock_sleep, mock_scrape_page):
    """Test that scraped content has correct structure"""
    mock_content = ScrapedContent(
        url="https://example.com",
        title="Test Page",
        content="Test content here",
        links=["https://example.com/link1"],
        metadata={"status_code": 200, "content-type": "text/html"},
        timestamp=datetime.now()
    )
    
    mock_scrape_page.return_value = mock_content
    
    scraper = GovernmentScraper("https://example.com", max_pages=1)
    results = scraper.scrape()
    
    assert len(results) == 1
    content = results[0]
    
    # Verify all required fields
    assert hasattr(content, 'url')
    assert hasattr(content, 'title')
    assert hasattr(content, 'content')
    assert hasattr(content, 'links')
    assert hasattr(content, 'metadata')
    assert hasattr(content, 'timestamp')
    
    assert content.url == "https://example.com"
    assert isinstance(content.links, list)
    assert isinstance(content.metadata, dict)
    assert isinstance(content.timestamp, datetime)


def test_govuk_site_detection():
    """Test that gov.uk sites are auto-detected"""
    scraper = GovernmentScraper("https://www.gov.uk/browse")
    assert scraper.site_type == "govuk"
    
    scraper = GovernmentScraper("https://data.gov.uk")
    assert scraper.site_type == "govuk"
    
    scraper = GovernmentScraper("https://example.com")
    assert scraper.site_type == "generic"


def test_govuk_scraper_real_page():
    """Test scraping a real gov.uk page (integration test)"""
    # Use a simple, stable gov.uk page
    scraper = GovernmentScraper("https://www.gov.uk/browse", max_pages=1)
    
    results = scraper.scrape()
    
    # Should successfully scrape at least one page
    assert len(results) >= 1
    
    content = results[0]
    
    # Verify gov.uk-specific extraction
    assert content.metadata.get("site_type") == "govuk"
    assert content.url == "https://www.gov.uk/browse"
    assert len(content.title) > 0
    assert len(content.content) > 0
    
    # Content should have structure (newlines preserved for gov.uk)
    assert "\n" in content.content or len(content.content) > 100
    
    # Should have timestamp
    assert isinstance(content.timestamp, datetime)


def test_govuk_content_structure():
    """Test that gov.uk content extraction preserves structure"""
    from bs4 import BeautifulSoup
    
    # Create HTML similar to gov.uk structure
    html = """
    <html>
        <head><title>Test GOV.UK Page</title></head>
        <body>
            <main>
                <h1>Main Heading</h1>
                <p>First paragraph with important information.</p>
                <h2>Subheading</h2>
                <ul>
                    <li>First list item</li>
                    <li>Second list item</li>
                </ul>
                <p>Another paragraph.</p>
            </main>
        </body>
    </html>
    """
    
    # Mock the response
    with patch('opengovcorpus.scraper.requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance
        
        scraper = GovernmentScraper("https://www.gov.uk/test", max_pages=1)
        scraper.session = mock_session_instance
        
        content = scraper._scrape_page("https://www.gov.uk/test")
        
        assert content is not None
        assert content.metadata.get("site_type") == "govuk"
        
        # Verify structure is preserved (newlines)
        assert "\n" in content.content
        
        # Verify list items are formatted
        assert "*" in content.content or "list item" in content.content.lower()
        
        # Verify main content is extracted (not nav/footer)
        assert "Main Heading" in content.content
        assert "First paragraph" in content.content


def test_govuk_link_filtering():
    """Test that gov.uk scraper filters /browse pages from links"""
    from bs4 import BeautifulSoup
    
    html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <main>
                <h1>Test Content</h1>
                <p>This is a test page with multiple links to verify filtering works correctly.</p>
                <p>We need enough content to pass the minimum content length check.</p>
                <ul>
                    <li>First item with more content</li>
                    <li>Second item with additional text</li>
                </ul>
                <a href="/content-page">Content Link</a>
                <a href="/browse/category">Browse Link</a>
                <a href="/another-page">Another Content</a>
            </main>
        </body>
    </html>
    """
    
    with patch('opengovcorpus.scraper.requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance
        
        scraper = GovernmentScraper("https://www.gov.uk/test", max_pages=1)
        scraper.session = mock_session_instance
        
        content = scraper._scrape_page("https://www.gov.uk/test")
        
        assert content is not None
        links = content.links
        
        # Should have content links
        assert len(links) > 0
        
        # Should NOT have /browse links
        browse_links = [link for link in links if "/browse" in link]
        assert len(browse_links) == 0, f"Found browse links: {browse_links}"


def test_govuk_main_element_extraction():
    """Test that gov.uk scraper targets <main> element"""
    from bs4 import BeautifulSoup
    
    html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation content should be ignored</nav>
            <main>
                <h1>Main Content</h1>
                <p>This is the actual content we want.</p>
            </main>
            <footer>Footer content should be ignored</footer>
        </body>
    </html>
    """
    
    with patch('opengovcorpus.scraper.requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance
        
        scraper = GovernmentScraper("https://www.gov.uk/test", max_pages=1)
        scraper.session = mock_session_instance
        
        content = scraper._scrape_page("https://www.gov.uk/test")
        
        assert content is not None
        # Should have main content
        assert "Main Content" in content.content
        assert "actual content" in content.content.lower()
        
        # Should NOT have nav/footer content
        assert "Navigation content" not in content.content
        assert "Footer content" not in content.content