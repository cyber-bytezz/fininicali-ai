"""
Base Agent Module.

This module defines the base class for all agents in the system.
"""
import logging
import uuid
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Optional unique identifier for the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = self.__class__.__name__
        logger.info(f"Initializing agent: {self.name} (ID: {self.agent_id})")
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed results
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Base implementation always returns True
        # Child classes should override this method for specific validation
        return True
    
    def log_activity(self, message: str, level: str = 'info'):
        """
        Log agent activity.
        
        Args:
            message: Log message
            level: Log level ('info', 'warning', 'error', 'debug')
        """
        if level == 'info':
            logger.info(f"[{self.name}] {message}")
        elif level == 'warning':
            logger.warning(f"[{self.name}] {message}")
        elif level == 'error':
            logger.error(f"[{self.name}] {message}")
        elif level == 'debug':
            logger.debug(f"[{self.name}] {message}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Dictionary containing agent status
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'type': self.__class__.__name__,
            'status': 'active'
        }
