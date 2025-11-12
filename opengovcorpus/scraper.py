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
    
    def __init__(self, base_url: str, max_pages: Optional[int] = None, 
                 site_type: Optional[str] = None):
        """
        Initialize scraper
        
        Args:
            base_url: Base URL to scrape
            max_pages: Maximum number of pages to scrape
            site_type: Optional site type ('govuk' or 'generic'). Auto-detected if None.
        """
        if not is_valid_url(base_url):
            raise ScraperError(f"Invalid URL: {base_url}")
        
        self.base_url = base_url
        self.domain = get_domain(base_url)
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.site_type = site_type or self._detect_site_type(base_url)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenGovCorpus/0.1.0 (Educational Purpose)'
        })
    
    def _detect_site_type(self, url: str) -> str:
        """
        Detect site type from URL
        
        Args:
            url: URL to check
            
        Returns:
            Site type string ('govuk' or 'generic')
        """
        if 'gov.uk' in url.lower():
            return 'govuk'
        return 'generic'
    
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
            
            # Use response.text for better encoding handling, fallback to content
            if hasattr(response, 'text') and isinstance(response.text, str) and response.text:
                html_content = response.text
            elif hasattr(response, 'content') and response.content:
                # Fallback for mocks or when text is not available
                if isinstance(response.content, bytes):
                    html_content = response.content.decode('utf-8', errors='ignore')
                else:
                    html_content = str(response.content)
            else:
                raise ScraperError(f"Unable to extract content from {url}")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract content based on site type
            if self.site_type == 'govuk':
                content, links = self._extract_govuk_content(soup, url)
            else:
                content, links = self._extract_generic_content(soup, url)
            
            # Skip if no content found (but allow short content for testing)
            if not content:
                return None
            
            # Metadata
            metadata = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'site_type': self.site_type
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
        except Exception as e:
            # Return None for processing errors (like notebook approach)
            # but still raise ScraperError for network/HTTP errors
            print(f"Error processing {url}: {e}")
            return None
    
    def _extract_govuk_content(self, soup: BeautifulSoup, url: str) -> tuple:
        """
        Extract content and links from gov.uk pages (like notebook approach)
        
        Args:
            soup: BeautifulSoup object
            url: URL being scraped
            
        Returns:
            Tuple of (content_text, links_list)
        """
        # Target main content area (like notebook)
        main_content = soup.find('main')
        if not main_content:
            # Fallback to body if no main element
            main_content = soup.find('body')
        
        if main_content:
            # Remove table of contents and navigation (like notebook)
            toc_classes_to_remove = ['gem-c-contents-list', 'contents', 'toc']
            for toc_class in toc_classes_to_remove:
                for element in main_content.find_all(class_=toc_class):
                    element.decompose()
            
            # Remove script, style, nav, footer, header
            for element in main_content(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract structured content (like notebook)
            content_lines = []
            for tag in main_content.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                text = ' '.join(tag.stripped_strings)
                if text:  # Only add non-empty text
                    if tag.name == 'li':
                        content_lines.append(f"* {text}")  # Format lists
                    else:
                        content_lines.append(text)
            content = '\n'.join(content_lines)
            
            # Extract links using gov.uk-specific selectors (like notebook)
            links = self._extract_govuk_links(main_content, url)
        else:
            # Fallback to generic extraction if no main element
            content, links = self._extract_generic_content(soup, url)
        
        return content, links
    
    def _extract_govuk_links(self, main_content, url: str) -> List[str]:
        """
        Extract links from gov.uk pages using specific selectors
        
        Args:
            main_content: BeautifulSoup element (main or body)
            url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        
        # Try gov.uk-specific selectors (like notebook)
        possible_selectors = [
            'main .gem-c-document-list a[href]',
            'main .browse-container .govuk-list a[href]',
            'main ul.govuk-list a[href]',
            'a[href]'  # Fallback
        ]
        
        link_elements = []
        for selector in possible_selectors:
            try:
                link_elements = main_content.select(selector)
                if link_elements:
                    break
            except:
                continue
        
        for link in link_elements:
            href = link.get('href', '')
            # Filter out browse pages (like notebook)
            if href.startswith('/') and not href.startswith('/browse'):
                absolute_link = urljoin(url, href)
                if is_valid_url(absolute_link) and get_domain(absolute_link) == self.domain:
                    links.append(absolute_link)
        
        return links
    
    def _extract_generic_content(self, soup: BeautifulSoup, url: str) -> tuple:
        """
        Extract content and links using generic approach (backward compatible)
        
        Args:
            soup: BeautifulSoup object
            url: URL being scraped
            
        Returns:
            Tuple of (content_text, links_list)
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract main content
        content = soup.get_text(separator=' ', strip=True)
        content = clean_text(content)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            try:
                href = link.get('href', '')
                if not href:
                    continue
                absolute_link = urljoin(url, href)
                if is_valid_url(absolute_link) and get_domain(absolute_link) == self.domain:
                    links.append(absolute_link)
            except Exception:
                # Skip invalid links
                continue
        
        return content, links


def scrape_website(url: str, max_pages: Optional[int] = None, 
                   site_type: Optional[str] = None) -> List[ScrapedContent]:
    """
    Scrape a government website
    
    Args:
        url: URL to scrape
        max_pages: Maximum number of pages
        site_type: Optional site type ('govuk' or 'generic'). Auto-detected if None.
        
    Returns:
        List of scraped content
    """
    scraper = GovernmentScraper(url, max_pages, site_type)
    return scraper.scrape()