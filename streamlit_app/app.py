"""
Streamlit App Module.

This module implements the Streamlit UI for the multi-agent finance assistant.
"""
import os
import logging
import json
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_URL = f"http://localhost:{os.getenv('PORT', '8000')}"

# Page configuration
st.set_page_config(
    page_title="Financial Market Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def get_api_url():
    """Get the API URL."""
    return API_URL

async def call_api(endpoint, method="GET", data=None, retry_count=3, retry_delay=1.5):
    """
    Call the API asynchronously with retry logic.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        API response or error dict
    """
    url = f"{get_api_url()}/{endpoint}"
    
    for attempt in range(retry_count):
        try:
            async with httpx.AsyncClient() as client:
                if method == "GET":
                    response = await client.get(url, timeout=30.0)
                elif method == "POST":
                    response = await client.post(url, json=data, timeout=60.0)
                else:
                    logger.error(f"Unsupported method: {method}")
                    return {"error": f"Unsupported method: {method}"}
                    
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error (attempt {attempt+1}/{retry_count}): {e}")
            if attempt == retry_count - 1:
                return {"error": f"HTTP error: {str(e)}"}
        except httpx.RequestError as e:
            logger.error(f"Request error (attempt {attempt+1}/{retry_count}): {e}")
            if attempt == retry_count - 1:
                return {"error": f"Cannot connect to API service. Please ensure the server is running at {url}"}
        except Exception as e:
            logger.error(f"Error calling API (attempt {attempt+1}/{retry_count}): {e}")
            if attempt == retry_count - 1:
                return {"error": f"Unexpected error: {str(e)}"}
        
        # Wait before retrying
        if attempt < retry_count - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    return {"error": "Maximum retry attempts exceeded"}

def run_async(func):
    """
    Run an async function from a synchronous context.
    
    Args:
        func: Async function to run
        
    Returns:
        Function result
    """
    return asyncio.run(func)

# App state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
    
# Add a session state for example queries
if 'example_query' not in st.session_state:
    st.session_state.example_query = ""
    
# Define a callback function for setting the example query
def set_example_query(query):
    st.session_state.example_query = query

# App layout
def main():
    """Main app function."""
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/financial-analytics.png", width=100)
        st.title("Financial Market Assistant")
        
        st.subheader("Settings")
        
        region = st.selectbox(
            "Region",
            ["Global", "Asia", "Europe", "US"],
            index=0
        )
        
        voice_output = st.checkbox("Enable voice output", value=False)
        
        # Market Data Sources
        st.subheader("Market Data Sources")
        
        data_sources = {
            "Yahoo Finance": True,
            "Alpha Vantage": False,
            "Financial News": True,
            "SEC Filings": False
        }
        
        for source, default in data_sources.items():
            st.checkbox(source, value=default, key=f"source_{source}")
        
        # Display portfolio
        st.subheader("Portfolio")
        
        # Mocked portfolio data for demo purposes
        portfolio = {
            "AAPL": 15.0,  # % of portfolio
            "MSFT": 12.0,
            "GOOGL": 10.0,
            "AMZN": 8.0,
            "META": 5.0,
            "TSMC": 7.0,
            "Samsung": 6.0,
            "SoftBank": 4.0,
            "Tencent": 5.0,
            "Alibaba": 4.0,
            "Sony": 3.0,
            "Nintendo": 2.0
        }
        
        # Create a DataFrame for display
        portfolio_df = pd.DataFrame(
            [(ticker, value) for ticker, value in portfolio.items()],
            columns=["Ticker", "Allocation (%)"]
        )
        
        st.dataframe(portfolio_df, hide_index=True)
        
        # About section
        st.subheader("About")
        st.write(
            "This multi-agent finance assistant provides market briefs "
            "by orchestrating specialized agents including API, Scraping, "
            "Retrieval, Analysis, Language, and Voice agents."
        )
        
        # API Status
        st.subheader("API Status")
        
        # Check API health
        if st.button("Check API Status"):
            with st.spinner("Checking API status..."):
                health_result = run_async(call_api("health"))
                
                if health_result and "error" not in health_result:
                    st.success(f"API is healthy: {health_result.get('timestamp')}")
                else:
                    st.error("API is not available")
    
    # Main content
    st.title("Morning Market Brief 📈")
    st.write("Ask about your portfolio, market trends, or earnings surprises.")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages
    st.subheader("Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display portfolio data if available
            if message["role"] == "assistant" and "portfolio_data" in message and message["portfolio_data"]:
                portfolio_data = message["portfolio_data"]
                
                # Display current vs previous allocation if available
                if "current_allocation" in portfolio_data and "previous_allocation" in portfolio_data:
                    st.info(f"Your Asia tech allocation is {portfolio_data['current_allocation']}% of AUM, "
                           f"{'up' if portfolio_data['change'] > 0 else 'down'} from {portfolio_data['previous_allocation']}% yesterday.")
                
                # Display pie chart if region data is available
                if "by_region" in portfolio_data:
                    region_data = portfolio_data["by_region"]
                    
                    # Create a DataFrame for the chart
                    df = pd.DataFrame(
                        [(region, allocation) for region, allocation in region_data.items()],
                        columns=["Region", "Allocation (%)"]
                    )
                    
                    # Create a pie chart
                    fig = px.pie(
                        df,
                        values="Allocation (%)",
                        names="Region",
                        title="Portfolio Allocation by Region",
                        color_discrete_sequence=px.colors.sequential.Plasma
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display sentiment data if available
                if "sentiment_data" in message and message["sentiment_data"]:
                    sentiment = message["sentiment_data"]
                    
                    # Use a more visually appealing display for sentiment
                    sentiment_value = 0.5  # neutral default
                    sentiment_color = "#FFD700"  # gold/yellow for neutral
                    sentiment_icon = "😐"  # neutral face
                    
                    if sentiment == "very positive" or sentiment == "bullish":
                        sentiment_value = 0.9
                        sentiment_color = "#00FF00"  # green
                        sentiment_icon = "🚀"  # rocket
                    elif sentiment == "positive" or sentiment == "somewhat bullish":
                        sentiment_value = 0.7
                        sentiment_color = "#90EE90"  # light green
                        sentiment_icon = "📈"  # chart up
                    elif sentiment == "neutral":
                        sentiment_value = 0.5
                        sentiment_color = "#FFD700"  # gold
                        sentiment_icon = "😐"  # neutral face
                    elif sentiment == "negative" or sentiment == "somewhat bearish":
                        sentiment_value = 0.3
                        sentiment_color = "#FFA07A"  # light salmon
                        sentiment_icon = "📉"  # chart down
                    elif sentiment == "very negative" or sentiment == "bearish":
                        sentiment_value = 0.1
                        sentiment_color = "#FF0000"  # red
                        sentiment_icon = "🐻"  # bear
                    
                    # Create a styled header for sentiment
                    st.markdown(f"<h3 style='color:{sentiment_color}'>{sentiment_icon} Market Sentiment: {sentiment.title()}</h3>", unsafe_allow_html=True)
                    
                    # Create a gauge chart for sentiment
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = sentiment_value * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Market Sentiment Index"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': sentiment_color},
                            'steps': [
                                {'range': [0, 20], 'color': "#FF6347"},  # tomato red
                                {'range': [20, 40], 'color': "#FFA07A"},  # light salmon
                                {'range': [40, 60], 'color': "#FFD700"},  # gold
                                {'range': [60, 80], 'color': "#90EE90"},  # light green
                                {'range': [80, 100], 'color': "#32CD32"}   # lime green
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': sentiment_value * 100
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display earnings data if available
                if "earnings_data" in message and message["earnings_data"]:
                    earnings_data = message["earnings_data"]
                    
                    # Extract positive and negative surprises
                    positive = earnings_data.get("positive_surprises", [])
                    negative = earnings_data.get("negative_surprises", [])
                    
                    if positive or negative:
                        st.markdown("### 📊 Earnings Surprises Analysis")
                        
                        # Create two columns for positive and negative surprises
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Positive Surprises")
                            
                            if positive:
                                # Create a more visually appealing display
                                df_positive = pd.DataFrame(positive)
                                if len(df_positive.columns) >= 3:  # Make sure we have the expected columns
                                    df_positive = df_positive[["symbol", "company", "surprise_pct"]]
                                    df_positive.columns = ["Symbol", "Company", "Surprise (%)"]
                                    
                                    # Format the surprise percentage with + sign and color
                                    df_positive["Surprise (%)"] = df_positive["Surprise (%)"].apply(
                                        lambda x: f"+{x:.1f}%" if isinstance(x, (int, float)) else x
                                    )
                                    
                                    # Create a bar chart of positive surprises
                                    fig = px.bar(
                                        df_positive,
                                        x="Symbol",
                                        y=[float(str(x).replace('%', '').replace('+', '')) for x in df_positive["Surprise (%)"]],
                                        color="Company",
                                        labels={"y": "Surprise (%)"},
                                        title="Positive Earnings Surprises",
                                        color_discrete_sequence=px.colors.sequential.Greens
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                # Also show the data in a table
                                st.dataframe(df_positive, hide_index=True, use_container_width=True)
                            else:
                                st.info("No positive earnings surprises reported.")
                        
                        with col2:
                            st.markdown("#### 📉 Negative Surprises")
                            
                            if negative:
                                # Create a more visually appealing display
                                df_negative = pd.DataFrame(negative)
                                if len(df_negative.columns) >= 3:  # Make sure we have the expected columns
                                    df_negative = df_negative[["symbol", "company", "surprise_pct"]]
                                    df_negative.columns = ["Symbol", "Company", "Surprise (%)"]
                                    
                                    # Format the surprise percentage
                                    df_negative["Surprise (%)"] = df_negative["Surprise (%)"].apply(
                                        lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
                                    )
                                    
                                    # Create a bar chart of negative surprises
                                    fig = px.bar(
                                        df_negative,
                                        x="Symbol",
                                        y=[abs(float(str(x).replace('%', ''))) for x in df_negative["Surprise (%)"]],
                                        color="Company",
                                        labels={"y": "Surprise (%) [Absolute Value]"},
                                        title="Negative Earnings Surprises",
                                        color_discrete_sequence=px.colors.sequential.Reds
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                # Also show the data in a table
                                st.dataframe(df_negative, hide_index=True, use_container_width=True)
                            else:
                                st.info("No negative earnings surprises reported.")
    
    # Input area
    query = st.text_input("Ask a question", key="query_input", value=st.session_state.example_query)
    
    # Clear the example query after it's been used
    if st.session_state.example_query:
        st.session_state.example_query = ""
    
    col1, col2 = st.columns([5, 1])
    
    # Chat input using Streamlit's native chat_input
    if prompt := st.chat_input("Ask a question about your portfolio or market trends"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Show assistant is thinking
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("Thinking...")
            
            # Call the orchestration API
            response = run_async(call_api(
                "orchestrate",
                method="POST",
                data={
                    "query": prompt,
                    "voice_input": None,
                    "voice_output": voice_output,
                    "region": region if region != "Global" else None
                }
            ))
            
            # Process the response
            if response and "error" not in response:
                # Get response content
                response_content = response.get("response", "I've processed your request but couldn't generate a proper response.")
                
                # Clear the thinking message
                thinking_placeholder.empty()
                
                # Display the actual response
                st.write(response_content)
                
                # Create assistant message object with all data
                assistant_msg = {
                    "role": "assistant",
                    "content": response_content,
                    "portfolio_data": response.get("portfolio_data"),
                    "sentiment_data": response.get("sentiment_data"),
                    "earnings_data": response.get("earnings_data")
                }
                
                # Display portfolio data if available
                if "portfolio_data" in response and response["portfolio_data"]:
                    portfolio_data = response["portfolio_data"]
                    
                    # Create an expandable section for detailed analytics
                    with st.expander("📊 Detailed Portfolio Analysis", expanded=True):
                        # Display current vs previous allocation if available
                        if "current_allocation" in portfolio_data and "previous_allocation" in portfolio_data:
                            # Create a metric with delta for better visualization
                            change = portfolio_data.get('change', portfolio_data['current_allocation'] - portfolio_data['previous_allocation'])
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label="Asia Tech Allocation (% of AUM)", 
                                    value=f"{portfolio_data['current_allocation']:.1f}%",
                                    delta=f"{change:.1f}%"
                                )
                            with col2:
                                st.metric(
                                    label="Previous Allocation", 
                                    value=f"{portfolio_data['previous_allocation']:.1f}%"
                                )
                        
                        # Display region data with both pie chart and bar chart for better visualization
                        if "by_region" in portfolio_data:
                            region_data = portfolio_data["by_region"]
                            
                            # Create a DataFrame for the charts
                            df = pd.DataFrame(
                                [(region, allocation) for region, allocation in region_data.items()],
                                columns=["Region", "Allocation (%)"]
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Create a pie chart
                                fig1 = px.pie(
                                    df,
                                    values="Allocation (%)",
                                    names="Region",
                                    title="Regional Allocation",
                                    color_discrete_sequence=px.colors.qualitative.Plotly
                                )
                                fig1.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Create a bar chart
                                fig2 = px.bar(
                                    df,
                                    x="Region",
                                    y="Allocation (%)",
                                    title="Regional Allocation",
                                    color="Region",
                                    color_discrete_sequence=px.colors.qualitative.Plotly
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                        
                        # Display sector allocation if available
                        if "by_sector" in portfolio_data:
                            sector_data = portfolio_data["by_sector"]
                            
                            # Create a DataFrame for the sector chart
                            df_sector = pd.DataFrame(
                                [(sector, allocation) for sector, allocation in sector_data.items()],
                                columns=["Sector", "Allocation (%)"]
                            )
                            
                            # Create a horizontal bar chart for sectors
                            fig3 = px.bar(
                                df_sector,
                                y="Sector",
                                x="Allocation (%)",
                                title="Sector Allocation",
                                orientation="h",
                                color="Sector",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                
                # Add message to history
                st.session_state.messages.append(assistant_msg)
            else:
                # Handle error response
                error_msg = "Sorry, I couldn't process your request."
                
                if response and "error" in response:
                    error_detail = response["error"]
                    if "Cannot connect to API service" in error_detail:
                        error_msg = ("I cannot connect to the backend services. Please make sure the FastAPI server "
                                    "is running. You can check this by running the following in a terminal:\n"
                                    "```\npython -m uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000\n```")
                    else:
                        error_msg = f"There was an error processing your request: {error_detail}"
                
                # Clear the thinking message and show error
                thinking_placeholder.empty()
                st.error(error_msg)
                
                # Add error message to history
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Default message if chat is empty
    if not st.session_state.messages:
        st.info(
            "👋 Hello! I'm your financial market assistant. You can ask me questions like:\n\n"
            "- What's our risk exposure in Asia tech stocks today?\n"
            "- Highlight any earnings surprises this week\n"
            "- How is market sentiment in Europe right now?\n"
            "- What are the top performing sectors today?"
        )
        
        # Example query buttons
        st.subheader("Try these examples:")
        
        example_queries = [
            "What's our risk exposure in Asia tech stocks today?",
            "Highlight any earnings surprises this week",
            "How is market sentiment in Europe right now?",
            "What are the top performing sectors today?"
        ]
        
        for query in example_queries:
            if st.button(query, on_click=set_example_query, args=(query,)):
                pass  # The callback will handle setting the query

if __name__ == "__main__":
    main()
