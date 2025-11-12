"""
Web scraping functionality for government websites
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime
import time

from .models import ScrapedContent
from .exceptions import ScraperError
from .utils import is_valid_url, get_domain, clean_text


class GovernmentScraper:
    """Scraper for government websites"""
    
    def __init__(self, base_url: str, max_pages: Optional[int] = None):
        """
        Initialize scraper
        
        Args:
            base_url: Base URL to scrape
            max_pages: Maximum number of pages to scrape
        """
        if not is_valid_url(base_url):
            raise ScraperError(f"Invalid URL: {base_url}")
        
        self.base_url = base_url
        self.domain = get_domain(base_url)
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenGovCorpus/0.1.0 (Educational Purpose)'
        })
    
    def scrape(self) -> List[ScrapedContent]:
        """
        Scrape the website
        
        Returns:
            List of scraped content
        """
        all_content = []
        urls_to_visit = [self.base_url]
        
        while urls_to_visit and (self.max_pages is None or len(all_content) < self.max_pages):
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            try:
                content = self._scrape_page(url)
                if content:
                    all_content.append(content)
                    self.visited_urls.add(url)
                    
                    # Add new links to visit
                    for link in content.links:
                        if link not in self.visited_urls and get_domain(link) == self.domain:
                            urls_to_visit.append(link)
                
                # Be polite - delay between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        return all_content
    
    def _scrape_page(self, url: str) -> Optional[ScrapedContent]:
        """
        Scrape a single page
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent or None
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract main content
            content = soup.get_text(separator=' ', strip=True)
            content = clean_text(content)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(url, link['href'])
                if is_valid_url(absolute_link):
                    links.append(absolute_link)
            
            # Metadata
            metadata = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', '')
            }
            
            return ScrapedContent(
                url=url,
                title=title_text,
                content=content,
                links=links,
                metadata=metadata,
                timestamp=datetime.now()
            )
            
        except requests.RequestException as e:
            raise ScraperError(f"Failed to scrape {url}: {e}")


def scrape_website(url: str, max_pages: Optional[int] = None) -> List[ScrapedContent]:
    """
    Scrape a government website
    
    Args:
        url: URL to scrape
        max_pages: Maximum number of pages
        
    Returns:
        List of scraped content
    """
    scraper = GovernmentScraper(url, max_pages)
    return scraper.scrape()