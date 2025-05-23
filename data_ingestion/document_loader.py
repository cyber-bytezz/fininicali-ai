"""
Document Loader Module.

This module handles loading and processing financial documents from various sources.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import tempfile

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Class for loading and processing financial documents."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the DocumentLoader.
        
        Args:
            cache_dir: Directory to cache downloaded documents
        """
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "financial_docs_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"DocumentLoader initialized with cache at {self.cache_dir}")
    
    def download_file(self, url: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Download a file from a URL and save it to the cache.
        
        Args:
            url: URL of the file to download
            filename: Optional filename to save as
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        try:
            if not filename:
                # Extract filename from URL
                filename = url.split('/')[-1]
                # Remove query parameters if any
                filename = filename.split('?')[0]
            
            # Create full path
            file_path = os.path.join(self.cache_dir, filename)
            
            # Check if file already exists in cache
            if os.path.exists(file_path):
                logger.info(f"File {filename} already exists in cache")
                return file_path
            
            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {url} to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def load_text_file(self, file_path: str) -> Optional[str]:
        """
        Load text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File contents as string or None if loading failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Successfully loaded text from {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error loading text from {file_path}: {str(e)}")
            return None
    
    def load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame or None if loading failed
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded CSV from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV from {file_path}: {str(e)}")
            return None
    
    def load_financial_report(self, url: str) -> Optional[str]:
        """
        Load and process a financial report from a URL.
        
        Args:
            url: URL of the financial report
            
        Returns:
            Processed text content or None if loading failed
        """
        try:
            # Download the file
            file_path = self.download_file(url)
            if not file_path:
                return None
            
            # Check file extension
            ext = Path(file_path).suffix.lower()
            
            if ext == '.txt':
                # Simple text file
                return self.load_text_file(file_path)
                
            elif ext == '.csv':
                # CSV file - convert to text summary
                df = self.load_csv_file(file_path)
                if df is not None:
                    summary = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n"
                    summary += f"Columns: {', '.join(df.columns)}\n"
                    summary += "Sample data:\n"
                    summary += df.head(5).to_string()
                    return summary
                return None
                
            elif ext in ['.pdf', '.htm', '.html']:
                # For PDF and HTML files, we'd use specialized parsers
                # This is a simplified implementation
                logger.warning(f"Specialized parsing for {ext} files not fully implemented")
                return f"Document at {url} requires specialized parsing."
                
            else:
                logger.warning(f"Unsupported file format: {ext}")
                return f"Unsupported file format: {ext}"
                
        except Exception as e:
            logger.error(f"Error processing financial report {url}: {str(e)}")
            return None
    
    def load_multiple_documents(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """
        Load multiple documents from a list of URLs.
        
        Args:
            urls: List of document URLs
            
        Returns:
            Dictionary mapping URLs to document contents
        """
        results = {}
        
        for url in urls:
            content = self.load_financial_report(url)
            results[url] = content
        
        logger.info(f"Loaded {len(results)} documents")
        return results
    
    def extract_sections(self, text: str, section_keywords: List[str]) -> Dict[str, str]:
        """
        Extract specific sections from a document based on keywords.
        
        Args:
            text: Document text
            section_keywords: List of section keywords to extract
            
        Returns:
            Dictionary mapping section names to section contents
        """
        if not text:
            return {}
            
        sections = {}
        
        for keyword in section_keywords:
            # Create a pattern to find sections
            pattern = rf"{keyword}[:\s]*(.*?)(?=\n\s*(?:{keyword}|$))"
            
            import re
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if matches:
                sections[keyword] = matches[0].strip()
            
        logger.info(f"Extracted {len(sections)} sections from document")
        return sections
