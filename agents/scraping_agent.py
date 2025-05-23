"""
Scraping Agent Module.

This module implements an agent that scrapes financial websites for data.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from agents.base_agent import BaseAgent
from data_ingestion.scraper import FinancialScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScrapingAgent(BaseAgent):
    """Agent for scraping financial websites."""
    
    def __init__(self, agent_id: Optional[str] = None, use_selenium: bool = False):
        """
        Initialize the scraping agent.
        
        Args:
            agent_id: Optional unique identifier for the agent
            use_selenium: Whether to use Selenium for dynamic content
        """
        super().__init__(agent_id)
        self.scraper = FinancialScraper(use_selenium=use_selenium)
        self.log_activity(f"Scraping Agent initialized (use_selenium={use_selenium})")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and scrape financial websites.
        
        Args:
            input_data: Dictionary containing request parameters
                - action: Type of data to scrape
                - params: Parameters for the specific action
            
        Returns:
            Dictionary containing scraped data
        """
        if not self.validate_input(input_data):
            self.log_activity("Invalid input data", "error")
            return {"error": "Invalid input data"}
        
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        self.log_activity(f"Processing action: {action} with params: {params}")
        
        try:
            if action == "get_financial_news":
                source = params.get("source", "yahoo")
                category = params.get("category", "markets")
                limit = params.get("limit", 10)
                
                data = self.scraper.get_financial_news(
                    source=source,
                    category=category,
                    limit=limit
                )
                
                return {
                    "data": data,
                    "metadata": {
                        "source": source,
                        "category": category,
                        "count": len(data)
                    }
                }
                
            elif action == "get_earnings_announcements":
                days = params.get("days", 7)
                region = params.get("region")
                
                data = self.scraper.get_earnings_announcements(
                    days=days,
                    region=region
                )
                
                return {
                    "data": data.to_dict(orient="records") if not data.empty else [],
                    "metadata": {
                        "days": days,
                        "region": region,
                        "count": len(data) if not data.empty else 0
                    }
                }
                
            elif action == "get_sec_filings":
                symbol = params.get("symbol")
                filing_type = params.get("filing_type", "10-K")
                limit = params.get("limit", 5)
                
                if not symbol:
                    return {"error": "Symbol is required"}
                
                data = self.scraper.get_sec_filings(
                    symbol=symbol,
                    filing_type=filing_type,
                    limit=limit
                )
                
                return {
                    "data": data,
                    "metadata": {
                        "symbol": symbol,
                        "filing_type": filing_type,
                        "count": len(data)
                    }
                }
                
            else:
                self.log_activity(f"Unknown action: {action}", "warning")
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.log_activity(f"Error processing action {action}: {str(e)}", "error")
            return {
                "error": str(e),
                "action": action,
                "params": params
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, dict):
            return False
            
        if "action" not in input_data:
            return False
            
        # Validate specific actions
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        if action == "get_sec_filings" and "symbol" not in params:
            return False
            
        return True
