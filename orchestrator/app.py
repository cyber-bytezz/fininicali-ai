"""
Orchestrator App Module.

This module implements the FastAPI application that coordinates the agents.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Financial Market Agent Orchestrator",
    description="API for the multi-agent finance assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response
class AgentRequest(BaseModel):
    """Model for agent request."""
    agent_type: str = Field(..., description="Type of agent to use")
    action: str = Field(..., description="Action to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")

class OrchestrationRequest(BaseModel):
    """Model for orchestration request."""
    query: str = Field(..., description="User query")
    voice_input: Optional[str] = Field(None, description="Path to voice input file")
    voice_output: bool = Field(False, description="Whether to generate voice output")
    region: Optional[str] = Field(None, description="Region filter")

class AgentResponse(BaseModel):
    """Model for agent response."""
    agent_id: str = Field(..., description="Agent ID")
    agent_type: str = Field(..., description="Type of agent")
    data: Dict[str, Any] = Field(..., description="Response data")
    timestamp: str = Field(..., description="Timestamp")

class OrchestrationResponse(BaseModel):
    """Model for orchestration response."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Text response")
    voice_output: Optional[str] = Field(None, description="Path to voice output file")
    agents_used: List[str] = Field(default_factory=list, description="List of agents used")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: str = Field(..., description="Timestamp")

# In-memory storage for agent instances and responses
agent_instances = {}
recent_responses = []

# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Market Agent Orchestrator API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/agent", response_model=AgentResponse)
async def call_agent(request: AgentRequest):
    """
    Call a specific agent directly.
    
    Args:
        request: AgentRequest model
        
    Returns:
        AgentResponse model
    """
    from orchestrator.agent_factory import get_agent_instance
    
    try:
        # Get agent instance
        agent = await get_agent_instance(request.agent_type)
        
        # Process request
        start_time = datetime.now()
        result = await agent.process({
            "action": request.action,
            "params": request.params
        })
        end_time = datetime.now()
        
        # Create response
        response = AgentResponse(
            agent_id=agent.agent_id,
            agent_type=request.agent_type,
            data=result,
            timestamp=datetime.now().isoformat()
        )
        
        # Log and return response
        logger.info(f"Agent {request.agent_type} processed {request.action} in {(end_time - start_time).total_seconds():.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error calling agent {request.agent_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(request: OrchestrationRequest, background_tasks: BackgroundTasks):
    """
    Orchestrate multiple agents to process a user query.
    
    Args:
        request: OrchestrationRequest model
        background_tasks: FastAPI BackgroundTasks
        
    Returns:
        OrchestrationResponse model
    """
    from orchestrator.orchestrator import process_market_brief
    
    try:
        # Start timing
        start_time = datetime.now()
        
        # Process request
        result = await process_market_brief(
            query=request.query,
            voice_input=request.voice_input,
            voice_output=request.voice_output,
            region=request.region
        )
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create response
        response = OrchestrationResponse(
            query=request.query,
            response=result["response"],
            voice_output=result.get("voice_output"),
            agents_used=result["agents_used"],
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Store recent response
        recent_responses.append(response.dict())
        if len(recent_responses) > 10:
            recent_responses.pop(0)
        
        # Log and return response
        logger.info(f"Orchestrated response for '{request.query}' in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error orchestrating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/recent")
async def get_recent_responses():
    """Get recent responses."""
    return recent_responses

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
