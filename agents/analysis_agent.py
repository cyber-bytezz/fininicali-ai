"""
Analysis Agent Module.

This module implements an agent that performs financial analysis on market data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """Agent for performing financial analysis on market data."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the analysis agent.
        
        Args:
            agent_id: Optional unique identifier for the agent
        """
        super().__init__(agent_id)
        self.log_activity("Analysis Agent initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and perform financial analysis.
        
        Args:
            input_data: Dictionary containing request parameters
                - action: Type of analysis to perform
                - params: Parameters for the specific analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.validate_input(input_data):
            self.log_activity("Invalid input data", "error")
            return {"error": "Invalid input data"}
        
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        self.log_activity(f"Processing action: {action} with params: {params}")
        
        try:
            if action == "analyze_stock_performance":
                stock_data = params.get("stock_data", [])
                period = params.get("period", "1mo")
                
                if not stock_data:
                    return {"error": "Stock data is required"}
                
                # Convert to DataFrame if necessary
                if isinstance(stock_data, list):
                    df = pd.DataFrame(stock_data)
                else:
                    df = stock_data
                
                analysis = self.analyze_stock_performance(df, period)
                return {
                    "analysis": analysis,
                    "metadata": {
                        "period": period
                    }
                }
                
            elif action == "analyze_portfolio":
                portfolio = params.get("portfolio", {})
                market_data = params.get("market_data", {})
                region = params.get("region")
                
                if not portfolio:
                    return {"error": "Portfolio data is required"}
                
                analysis = self.analyze_portfolio(portfolio, market_data, region)
                return {
                    "analysis": analysis,
                    "metadata": {
                        "region": region
                    }
                }
                
            elif action == "analyze_earnings_surprise":
                earnings_data = params.get("earnings_data", [])
                
                if not earnings_data:
                    return {"error": "Earnings data is required"}
                
                analysis = self.analyze_earnings_surprise(earnings_data)
                return {
                    "analysis": analysis,
                    "metadata": {
                        "count": len(earnings_data)
                    }
                }
                
            elif action == "analyze_market_sentiment":
                sentiment_data = params.get("sentiment_data", {})
                region = params.get("region")
                
                if not sentiment_data:
                    return {"error": "Sentiment data is required"}
                
                analysis = self.analyze_market_sentiment(sentiment_data, region)
                return {
                    "analysis": analysis,
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
    
    def analyze_stock_performance(
        self, 
        stock_data: pd.DataFrame, 
        period: str = "1mo"
    ) -> Dict[str, Any]:
        """
        Analyze stock performance metrics.
        
        Args:
            stock_data: DataFrame containing stock price data
            period: Time period for analysis
            
        Returns:
            Dictionary containing performance metrics
        """
        # Ensure we have required columns
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        
        if missing_cols:
            self.log_activity(f"Missing columns: {missing_cols}", "warning")
            # Use available columns only
        
        analysis = {}
        
        # Calculate return metrics if we have Close prices
        if 'Close' in stock_data.columns:
            # Calculate daily returns
            stock_data['daily_return'] = stock_data['Close'].pct_change() * 100
            
            # Calculate cumulative return
            first_close = stock_data['Close'].iloc[0]
            last_close = stock_data['Close'].iloc[-1]
            cumulative_return = ((last_close / first_close) - 1) * 100
            
            # Calculate volatility
            volatility = stock_data['daily_return'].std() * np.sqrt(252)  # Annualized
            
            # Calculate max drawdown
            peak = stock_data['Close'].cummax()
            drawdown = (stock_data['Close'] / peak - 1) * 100
            max_drawdown = drawdown.min()
            
            analysis['return_metrics'] = {
                'current_price': float(last_close),
                'price_change': float(last_close - first_close),
                'percent_change': float(cumulative_return),
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown)
            }
        
        # Calculate volume metrics if we have Volume
        if 'Volume' in stock_data.columns:
            avg_volume = stock_data['Volume'].mean()
            last_volume = stock_data['Volume'].iloc[-1]
            volume_change = ((last_volume / avg_volume) - 1) * 100
            
            analysis['volume_metrics'] = {
                'average_volume': float(avg_volume),
                'last_volume': float(last_volume),
                'volume_change': float(volume_change)
            }
        
        # Calculate technical indicators if we have OHLC
        if all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Calculate simple moving averages
            stock_data['sma_5'] = stock_data['Close'].rolling(window=5).mean()
            stock_data['sma_20'] = stock_data['Close'].rolling(window=20).mean()
            
            # RSI (Relative Strength Index)
            delta = stock_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            stock_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            stock_data['ema_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
            stock_data['ema_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['macd'] = stock_data['ema_12'] - stock_data['ema_26']
            stock_data['macd_signal'] = stock_data['macd'].ewm(span=9, adjust=False).mean()
            
            last_row = stock_data.iloc[-1]
            
            analysis['technical_indicators'] = {
                'sma_5': float(last_row['sma_5']) if not pd.isna(last_row['sma_5']) else None,
                'sma_20': float(last_row['sma_20']) if not pd.isna(last_row['sma_20']) else None,
                'rsi': float(last_row['rsi']) if not pd.isna(last_row['rsi']) else None,
                'macd': float(last_row['macd']) if not pd.isna(last_row['macd']) else None,
                'macd_signal': float(last_row['macd_signal']) if not pd.isna(last_row['macd_signal']) else None,
                'above_sma_20': bool(last_row['Close'] > last_row['sma_20']) if not pd.isna(last_row['sma_20']) else None
            }
        
        # Calculate trend analysis
        if 'Close' in stock_data.columns and len(stock_data) > 1:
            # Determine trend
            short_term_change = (
                (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-min(5, len(stock_data))]) - 1
            ) * 100
            
            mid_term_change = (
                (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1
            ) * 100
            
            # Determine trend direction
            if short_term_change > 2:
                short_term_trend = "strong bullish"
            elif short_term_change > 0.5:
                short_term_trend = "bullish"
            elif short_term_change < -2:
                short_term_trend = "strong bearish"
            elif short_term_change < -0.5:
                short_term_trend = "bearish"
            else:
                short_term_trend = "neutral"
            
            if mid_term_change > 5:
                mid_term_trend = "strong bullish"
            elif mid_term_change > 1:
                mid_term_trend = "bullish"
            elif mid_term_change < -5:
                mid_term_trend = "strong bearish"
            elif mid_term_change < -1:
                mid_term_trend = "bearish"
            else:
                mid_term_trend = "neutral"
            
            analysis['trend_analysis'] = {
                'short_term_change': float(short_term_change),
                'mid_term_change': float(mid_term_change),
                'short_term_trend': short_term_trend,
                'mid_term_trend': mid_term_trend
            }
        
        return analysis
    
    def analyze_portfolio(
        self, 
        portfolio: Dict[str, float], 
        market_data: Dict[str, Any],
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze portfolio performance and risk metrics.
        
        Args:
            portfolio: Dictionary mapping ticker symbols to weights or values
            market_data: Dictionary containing market data for analysis
            region: Optional region filter
            
        Returns:
            Dictionary containing portfolio analysis
        """
        analysis = {
            'portfolio_summary': {},
            'risk_metrics': {},
            'exposure_analysis': {},
            'recommendations': []
        }
        
        # Extract portfolio metrics
        total_value = sum(portfolio.values())
        ticker_count = len(portfolio)
        
        analysis['portfolio_summary'] = {
            'total_value': total_value,
            'ticker_count': ticker_count
        }
        
        # Extract exposure analysis if available
        if 'by_region' in market_data and 'by_sector' in market_data:
            analysis['exposure_analysis'] = {
                'region_exposure': market_data['by_region'],
                'sector_exposure': market_data['by_sector']
            }
            
            # Filter to specific region if requested
            if region:
                region_exposure = {k: v for k, v in market_data['by_region'].items() 
                                 if region.lower() in k.lower()}
                analysis['exposure_analysis']['filtered_region_exposure'] = region_exposure
                
                # Calculate total exposure to the region
                total_region_exposure = sum(region_exposure.values())
                analysis['exposure_analysis']['total_region_exposure'] = total_region_exposure
                
                # Generate recommendations based on exposure
                if total_region_exposure > 30:
                    analysis['recommendations'].append(
                        f"High exposure ({total_region_exposure:.1f}%) to {region} region. "
                        f"Consider diversifying to reduce concentration risk."
                    )
                elif total_region_exposure < 10:
                    analysis['recommendations'].append(
                        f"Low exposure ({total_region_exposure:.1f}%) to {region} region. "
                        f"Consider increasing allocation if you want more exposure to this region."
                    )
        
        # Calculate concentration risk
        max_allocation = max(portfolio.values()) / total_value * 100
        top_holdings = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_concentration = sum(weight for _, weight in top_holdings) / total_value * 100
        
        analysis['risk_metrics'] = {
            'max_allocation': float(max_allocation),
            'top_3_concentration': float(top_3_concentration)
        }
        
        # Add recommendations based on concentration
        if max_allocation > 15:
            max_ticker, max_value = max(portfolio.items(), key=lambda x: x[1])
            analysis['recommendations'].append(
                f"High concentration in {max_ticker} ({max_allocation:.1f}%). "
                f"Consider reducing position to limit single-stock risk."
            )
            
        if top_3_concentration > 50:
            analysis['recommendations'].append(
                f"Top 3 holdings represent {top_3_concentration:.1f}% of the portfolio. "
                f"Consider diversifying to reduce concentration risk."
            )
        
        return analysis
    
    def analyze_earnings_surprise(self, earnings_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze earnings surprises from a list of earnings announcements.
        
        Args:
            earnings_data: List of dictionaries containing earnings data
            
        Returns:
            Dictionary containing earnings surprise analysis
        """
        analysis = {
            'positive_surprises': [],
            'negative_surprises': [],
            'summary': {}
        }
        
        positive_count = 0
        negative_count = 0
        total_analyzable = 0
        
        for item in earnings_data:
            # Skip if we don't have both estimated and reported EPS
            if 'eps_estimate' not in item or 'reported_eps' not in item:
                continue
                
            # Skip if either value is empty or not convertible to float
            try:
                eps_estimate = float(item['eps_estimate'].replace('$', '').strip())
                reported_eps = float(item['reported_eps'].replace('$', '').strip())
            except (ValueError, AttributeError):
                continue
                
            total_analyzable += 1
            
            # Calculate surprise percentage
            if eps_estimate != 0:
                surprise_pct = ((reported_eps - eps_estimate) / abs(eps_estimate)) * 100
            else:
                # Avoid division by zero
                surprise_pct = 0 if reported_eps == 0 else 100
                
            # Determine if it's a positive or negative surprise
            if surprise_pct > 1:  # More than 1% positive surprise
                positive_count += 1
                if 'symbol' in item and 'company' in item:
                    analysis['positive_surprises'].append({
                        'symbol': item['symbol'],
                        'company': item['company'],
                        'eps_estimate': eps_estimate,
                        'reported_eps': reported_eps,
                        'surprise_pct': float(surprise_pct)
                    })
            elif surprise_pct < -1:  # More than 1% negative surprise
                negative_count += 1
                if 'symbol' in item and 'company' in item:
                    analysis['negative_surprises'].append({
                        'symbol': item['symbol'],
                        'company': item['company'],
                        'eps_estimate': eps_estimate,
                        'reported_eps': reported_eps,
                        'surprise_pct': float(surprise_pct)
                    })
        
        # Sort surprises by magnitude
        analysis['positive_surprises'] = sorted(
            analysis['positive_surprises'], 
            key=lambda x: x['surprise_pct'], 
            reverse=True
        )[:5]  # Limit to top 5
        
        analysis['negative_surprises'] = sorted(
            analysis['negative_surprises'], 
            key=lambda x: x['surprise_pct']
        )[:5]  # Limit to top 5 (most negative)
        
        # Calculate summary metrics
        if total_analyzable > 0:
            positive_pct = (positive_count / total_analyzable) * 100
            negative_pct = (negative_count / total_analyzable) * 100
            
            analysis['summary'] = {
                'total_analyzed': total_analyzable,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_pct': float(positive_pct),
                'negative_pct': float(negative_pct)
            }
            
            # Determine overall earnings sentiment
            if positive_pct > 70:
                analysis['summary']['sentiment'] = "very positive"
            elif positive_pct > 55:
                analysis['summary']['sentiment'] = "positive"
            elif negative_pct > 70:
                analysis['summary']['sentiment'] = "very negative"
            elif negative_pct > 55:
                analysis['summary']['sentiment'] = "negative"
            else:
                analysis['summary']['sentiment'] = "neutral"
        
        return analysis
    
    def analyze_market_sentiment(
        self, 
        sentiment_data: Dict[str, Any],
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment indicators.
        
        Args:
            sentiment_data: Dictionary containing sentiment data
            region: Optional region filter
            
        Returns:
            Dictionary containing sentiment analysis
        """
        analysis = {
            'sentiment_summary': {},
            'key_indicators': [],
            'notable_moves': []
        }
        
        # Extract overall sentiment if available
        if 'overall' in sentiment_data:
            analysis['sentiment_summary']['overall'] = sentiment_data['overall']
        
        # Extract trend if available
        if 'trend' in sentiment_data:
            analysis['sentiment_summary']['trend'] = sentiment_data['trend']
        
        # Extract volatility if available
        if 'volatility' in sentiment_data:
            analysis['sentiment_summary']['volatility'] = sentiment_data['volatility']
        
        # Extract region-specific context if region is provided
        if region:
            analysis['sentiment_summary']['region'] = region
        
        # Analyze indices if available
        if 'indices' in sentiment_data:
            # Count sentiment by category
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            for index, data in sentiment_data['indices'].items():
                if 'sentiment' in data:
                    sentiment = data['sentiment']
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                # Add notable moves (significant daily changes)
                if 'daily_change' in data and abs(data['daily_change']) > 1.5:
                    analysis['notable_moves'].append({
                        'index': index,
                        'daily_change': data['daily_change'],
                        'direction': 'up' if data['daily_change'] > 0 else 'down'
                    })
            
            # Determine consensus sentiment
            total = sum(sentiment_counts.values())
            if total > 0:
                for sentiment, count in sentiment_counts.items():
                    analysis['sentiment_summary'][f'{sentiment}_pct'] = (count / total) * 100
                
                # Add key indicators
                if sentiment_counts['positive'] > sentiment_counts['negative'] * 2:
                    analysis['key_indicators'].append("Most indices showing positive momentum")
                elif sentiment_counts['negative'] > sentiment_counts['positive'] * 2:
                    analysis['key_indicators'].append("Most indices showing negative momentum")
                
                # Check for divergence
                if (sentiment_counts['positive'] > 0 and sentiment_counts['negative'] > 0 and
                        abs(sentiment_counts['positive'] - sentiment_counts['negative']) <= 1):
                    analysis['key_indicators'].append("Mixed signals across indices indicating uncertainty")
        
        # Additional factors that influence sentiment
        additional_factors = []
        
        # Add yield information if available
        if 'yields' in sentiment_data:
            for yield_name, yield_data in sentiment_data['yields'].items():
                if 'change' in yield_data and abs(yield_data['change']) > 0.05:
                    direction = 'rising' if yield_data['change'] > 0 else 'falling'
                    additional_factors.append(f"{yield_name} yields {direction}")
        
        # Add additional factors to key indicators
        if additional_factors:
            analysis['key_indicators'].extend(additional_factors)
        
        # Sort notable moves by magnitude
        analysis['notable_moves'] = sorted(
            analysis['notable_moves'], 
            key=lambda x: abs(x['daily_change']), 
            reverse=True
        )
        
        return analysis
