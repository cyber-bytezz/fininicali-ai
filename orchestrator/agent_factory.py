"""
Agent Factory Module.

This module provides functions for creating and managing agent instances.
"""
import logging
from typing import Dict, Any, Optional

# Import agent classes
from agents.api_agent import APIAgent
from agents.scraping_agent import ScrapingAgent
from agents.retriever_agent import RetrieverAgent
from agents.analysis_agent import AnalysisAgent
from agents.language_agent import LanguageAgent
from agents.voice_agent import VoiceAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dictionary to store agent instances
agent_instances = {}

async def get_agent_instance(agent_type: str, **kwargs) -> Any:
    """
    Get or create an agent instance.
    
    Args:
        agent_type: Type of agent to get or create
        **kwargs: Additional parameters for agent initialization
        
    Returns:
        Agent instance
    """
    # Check if we already have an instance of this agent type
    if agent_type in agent_instances:
        logger.info(f"Using existing {agent_type} instance")
        return agent_instances[agent_type]
    
    # Create a new agent instance based on the type
    logger.info(f"Creating new {agent_type} instance")
    
    if agent_type == "api":
        agent = APIAgent(**kwargs)
    elif agent_type == "scraping":
        agent = ScrapingAgent(**kwargs)
    elif agent_type == "retriever":
        agent = RetrieverAgent(**kwargs)
    elif agent_type == "analysis":
        agent = AnalysisAgent(**kwargs)
    elif agent_type == "language":
        agent = LanguageAgent(**kwargs)
    elif agent_type == "voice":
        agent = VoiceAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Store the agent instance
    agent_instances[agent_type] = agent
    
    return agent

async def clear_agent_instances():
    """Clear all agent instances."""
    global agent_instances
    
    for agent_type, agent in agent_instances.items():
        logger.info(f"Clearing {agent_type} instance")
    
    agent_instances = {}
    logger.info("All agent instances cleared")

async def get_agent_status() -> Dict[str, Any]:
    """
    Get status of all agent instances.
    
    Returns:
        Dictionary mapping agent types to status information
    """
    status = {}
    
    for agent_type, agent in agent_instances.items():
        status[agent_type] = agent.get_status()
    
    return status
