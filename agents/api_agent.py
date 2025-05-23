"""
API Agent Module.

This module implements an agent that fetches data from financial APIs.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from agents.base_agent import BaseAgent
from data_ingestion.api_data import MarketDataAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIAgent(BaseAgent):
    """Agent for fetching data from financial APIs."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the API agent.
        
        Args:
            agent_id: Optional unique identifier for the agent
        """
        super().__init__(agent_id)
        self.market_data_api = MarketDataAPI()
        self.log_activity("API Agent initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and fetch financial data from APIs.
        
        Args:
            input_data: Dictionary containing request parameters
                - action: Type of data to fetch
                - params: Parameters for the specific action
            
        Returns:
            Dictionary containing fetched data
        """
        if not self.validate_input(input_data):
            self.log_activity("Invalid input data", "error")
            return {"error": "Invalid input data"}
        
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        self.log_activity(f"Processing action: {action} with params: {params}")
        
        try:
            if action == "get_stock_data":
                symbol = params.get("symbol")
                source = params.get("source", "yfinance")
                period = params.get("period", "1d")
                
                if not symbol:
                    return {"error": "Symbol is required"}
                
                data = self.market_data_api.get_stock_data(
                    symbol=symbol,
                    source=source,
                    period=period
                )
                
                # Convert DataFrame to dict for JSON serialization
                return {
                    "data": data.to_dict(orient="records"),
                    "metadata": {
                        "symbol": symbol,
                        "source": source,
                        "period": period
                    }
                }
                
            elif action == "get_company_overview":
                symbol = params.get("symbol")
                
                if not symbol:
                    return {"error": "Symbol is required"}
                
                data = self.market_data_api.get_company_overview(symbol=symbol)
                return {
                    "data": data,
                    "metadata": {
                        "symbol": symbol
                    }
                }
                
            elif action == "get_earnings":
                symbol = params.get("symbol")
                
                if not symbol:
                    return {"error": "Symbol is required"}
                
                data = self.market_data_api.get_earnings(symbol=symbol)
                return {
                    "data": data.to_dict(orient="records") if not data.empty else [],
                    "metadata": {
                        "symbol": symbol
                    }
                }
                
            elif action == "get_sector_performance":
                data = self.market_data_api.get_sector_performance()
                return {
                    "data": data.to_dict(orient="records") if not data.empty else [],
                    "metadata": {}
                }
                
            elif action == "calculate_portfolio_exposure":
                portfolio = params.get("portfolio", {})
                region = params.get("region")
                
                if not portfolio:
                    return {"error": "Portfolio is required"}
                
                data = self.market_data_api.calculate_portfolio_exposure(
                    portfolio=portfolio,
                    region=region
                )
                return {
                    "data": data,
                    "metadata": {
                        "region": region
                    }
                }
                
            elif action == "get_market_sentiment":
                region = params.get("region")
                data = self.market_data_api.get_market_sentiment(region=region)
                return {
                    "data": data,
                    "metadata": {
                        "region": region
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
            
        return True
