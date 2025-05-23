"""
Web Scraping Module.

This module handles web scraping of financial news, earnings reports, and SEC filings.
It uses BeautifulSoup for static content and optionally Selenium for dynamic content.
"""
import logging
import re
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialScraper:
    """Class for scraping financial data from various websites."""
    
    def __init__(self, use_selenium: bool = False):
        """
        Initialize the FinancialScraper.
        
        Args:
            use_selenium: Whether to use Selenium for dynamic content
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.use_selenium = use_selenium
        
        # Initialize Selenium if needed
        if self.use_selenium:
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                
                self.driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
                logger.info("Selenium WebDriver initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Selenium: {str(e)}")
                self.use_selenium = False
                logger.warning("Falling back to requests-only mode")
        
        logger.info("FinancialScraper initialized")
    
    def __del__(self):
        """Clean up resources."""
        if self.use_selenium and hasattr(self, 'driver'):
            try:
                self.driver.quit()
                logger.info("Selenium WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing Selenium WebDriver: {str(e)}")
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a web page and return a BeautifulSoup object.
        
        Args:
            url: The URL to fetch
            
        Returns:
            BeautifulSoup object or None if an error occurs
        """
        try:
            if self.use_selenium:
                self.driver.get(url)
                time.sleep(2)  # Allow time for JavaScript to execute
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
            else:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
            
            logger.info(f"Successfully fetched {url}")
            return soup
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def get_financial_news(
        self, 
        source: str = 'yahoo',
        category: str = 'markets',
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Scrape financial news articles.
        
        Args:
            source: News source ('yahoo', 'bloomberg', 'reuters')
            category: News category ('markets', 'stocks', 'economy')
            limit: Maximum number of articles to return
            
        Returns:
            List of dictionaries containing article data
        """
        articles = []
        
        try:
            if source.lower() == 'yahoo':
                url = f"https://finance.yahoo.com/{category}"
                soup = self.fetch_page(url)
                
                if soup:
                    # Find news articles
                    article_tags = soup.find_all('h3', class_='Mb(5px)')
                    
                    for i, article_tag in enumerate(article_tags):
                        if i >= limit:
                            break
                            
                        link_tag = article_tag.find('a')
                        if link_tag:
                            title = link_tag.text.strip()
                            link = link_tag.get('href')
                            
                            # Make sure the link is absolute
                            if link and link.startswith('/'):
                                link = f"https://finance.yahoo.com{link}"
                            
                            # Get article date and summary if available
                            date = datetime.now().strftime('%Y-%m-%d')
                            summary = ""
                            
                            p_tag = article_tag.find_next('p')
                            if p_tag:
                                summary = p_tag.text.strip()
                            
                            articles.append({
                                'title': title,
                                'link': link,
                                'date': date,
                                'source': 'Yahoo Finance',
                                'summary': summary
                            })
            
            elif source.lower() == 'reuters':
                url = f"https://www.reuters.com/business/{category}"
                soup = self.fetch_page(url)
                
                if soup:
                    # Find news articles
                    article_tags = soup.find_all('article')
                    
                    for i, article_tag in enumerate(article_tags):
                        if i >= limit:
                            break
                            
                        title_tag = article_tag.find('h3')
                        if not title_tag:
                            continue
                            
                        title = title_tag.text.strip()
                        
                        link_tag = title_tag.find('a') or article_tag.find('a')
                        if not link_tag:
                            continue
                            
                        link = link_tag.get('href')
                        
                        # Make sure the link is absolute
                        if link and link.startswith('/'):
                            link = f"https://www.reuters.com{link}"
                        
                        # Get article date and summary if available
                        date = datetime.now().strftime('%Y-%m-%d')
                        summary = ""
                        
                        time_tag = article_tag.find('time')
                        if time_tag:
                            date_str = time_tag.get('datetime')
                            if date_str:
                                try:
                                    date = datetime.fromisoformat(date_str).strftime('%Y-%m-%d')
                                except ValueError:
                                    pass
                        
                        p_tag = article_tag.find('p')
                        if p_tag:
                            summary = p_tag.text.strip()
                        
                        articles.append({
                            'title': title,
                            'link': link,
                            'date': date,
                            'source': 'Reuters',
                            'summary': summary
                        })
            
            # Add more sources as needed
            
            logger.info(f"Successfully scraped {len(articles)} articles from {source}")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping news from {source}: {str(e)}")
            return articles
    
    def get_earnings_announcements(
        self,
        days: int = 7,
        region: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Scrape upcoming and recent earnings announcements.
        
        Args:
            days: Number of days to look ahead/behind
            region: Optional region filter
            
        Returns:
            DataFrame containing earnings announcement data
        """
        try:
            # Yahoo Finance earnings calendar
            today = datetime.now()
            start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = (today + timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"https://finance.yahoo.com/calendar/earnings?from={start_date}&to={end_date}"
            soup = self.fetch_page(url)
            
            earnings_data = []
            
            if soup:
                table = soup.find('table')
                if table:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 5:
                            symbol = cells[0].text.strip()
                            company = cells[1].text.strip()
                            date = cells[2].text.strip()
                            time = cells[3].text.strip()
                            eps_estimate = cells[4].text.strip()
                            
                            # Extract reported EPS if available
                            reported_eps = ""
                            if len(cells) >= 6:
                                reported_eps = cells[5].text.strip()
                            
                            # Check if we need to filter by region
                            include = True
                            if region:
                                # This would require additional info about the company's region
                                # For now, we'll include all and let the caller filter later
                                pass
                            
                            if include:
                                earnings_data.append({
                                    'symbol': symbol,
                                    'company': company,
                                    'date': date,
                                    'time': time,
                                    'eps_estimate': eps_estimate,
                                    'reported_eps': reported_eps
                                })
            
            df = pd.DataFrame(earnings_data)
            logger.info(f"Successfully scraped {len(earnings_data)} earnings announcements")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping earnings announcements: {str(e)}")
            return pd.DataFrame()
    
    def get_sec_filings(
        self,
        symbol: str,
        filing_type: str = '10-K',
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Scrape SEC filings for a specific company.
        
        Args:
            symbol: The stock ticker symbol
            filing_type: Type of filing ('10-K', '10-Q', '8-K', etc.)
            limit: Maximum number of filings to return
            
        Returns:
            List of dictionaries containing filing data
        """
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}&type={filing_type}&count={limit}"
            soup = self.fetch_page(url)
            
            filings = []
            
            if soup:
                # Find the filing table
                table = soup.find('table', class_='tableFile2')
                if table:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            filing_type = cells[0].text.strip()
                            date = cells[1].text.strip()
                            
                            # Find the link to the filing
                            link = ""
                            for a_tag in cells[1].find_all('a'):
                                if 'href' in a_tag.attrs:
                                    link = f"https://www.sec.gov{a_tag['href']}"
                                    break
                            
                            description = cells[2].text.strip()
                            
                            filings.append({
                                'filing_type': filing_type,
                                'date': date,
                                'description': description,
                                'link': link
                            })
                            
                            if len(filings) >= limit:
                                break
            
            logger.info(f"Successfully scraped {len(filings)} SEC filings for {symbol}")
            return filings
            
        except Exception as e:
            logger.error(f"Error scraping SEC filings for {symbol}: {str(e)}")
            return []
