"""
BDD-style tests for the agent implementations.

This module uses pytest and implements tests following
the Semantic Seed Venture Studio Coding Standards V2.0 with
a focus on Behavior-Driven Development (BDD).
"""
import pytest
import asyncio
import json
import os
from pathlib import Path

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from agents.api_agent import APIAgent
from agents.language_agent import LanguageAgent
from agents.analysis_agent import AnalysisAgent

# BDD-style test for API Agent
class TestAPIAgent:
    """Tests for the API Agent following BDD style."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_agent = APIAgent()
    
    @pytest.mark.asyncio
    async def test_portfolio_exposure_calculation(self):
        """
        Given a portfolio with Asia tech stocks
        When I calculate the portfolio exposure
        Then I should get proper regional allocation percentages
        """
        # Given
        portfolio = {
            "TSMC": 7.0,
            "Samsung": 6.0,
            "SoftBank": 4.0,
            "Tencent": 5.0,
            "Alibaba": 4.0
        }
        
        # When
        result = await self.api_agent.process({
            "action": "calculate_portfolio_exposure",
            "params": {
                "portfolio": portfolio,
                "region": "Asia"
            }
        })
        
        # Then
        assert "data" in result
        assert "by_region" in result["data"]
        assert result["data"]["by_region"]
        # At least one Asia region should be in the results
        assert any("Asia" in region or "China" in region or "Japan" in region or "Taiwan" in region 
                  or "Korea" in region for region in result["data"]["by_region"])

# BDD-style test for Language Agent
class TestLanguageAgent:
    """Tests for the Language Agent following BDD style."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.language_agent = LanguageAgent()
    
    @pytest.mark.asyncio
    async def test_generate_text(self):
        """
        Given a prompt about financial markets
        When I generate text
        Then I should get a non-empty response
        """
        # Given
        prompt = "What are the key factors affecting stock markets today?"
        
        # Skip test if no API key
        if not os.getenv("TOGETHER_API_KEY"):
            pytest.skip("No TOGETHER_API_KEY environment variable found")
        
        # When
        result = await self.language_agent.process({
            "action": "generate_text",
            "params": {
                "prompt": prompt,
                "max_tokens": 50
            }
        })
        
        # Then
        assert "response" in result
        assert len(result["response"]) > 0

# BDD-style test for Analysis Agent
class TestAnalysisAgent:
    """Tests for the Analysis Agent following BDD style."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analysis_agent = AnalysisAgent()
    
    @pytest.mark.asyncio
    async def test_earnings_surprise_analysis(self):
        """
        Given earnings data with surprises
        When I analyze earnings surprises
        Then I should get categorized positive and negative surprises
        """
        # Given
        earnings_data = [
            {
                "symbol": "AAPL",
                "company": "Apple Inc.",
                "eps_estimate": "1.50",
                "reported_eps": "1.65"
            },
            {
                "symbol": "MSFT",
                "company": "Microsoft Corp.",
                "eps_estimate": "2.30",
                "reported_eps": "2.35"
            },
            {
                "symbol": "GOOGL",
                "company": "Alphabet Inc.",
                "eps_estimate": "1.40",
                "reported_eps": "1.35"
            }
        ]
        
        # When
        result = await self.analysis_agent.process({
            "action": "analyze_earnings_surprise",
            "params": {
                "earnings_data": earnings_data
            }
        })
        
        # Then
        assert "analysis" in result
        assert "positive_surprises" in result["analysis"]
        assert "negative_surprises" in result["analysis"]
        assert any(surprise["symbol"] == "AAPL" for surprise in result["analysis"]["positive_surprises"])
        assert any(surprise["symbol"] == "GOOGL" for surprise in result["analysis"]["negative_surprises"])


if __name__ == "__main__":
    pytest.main()
