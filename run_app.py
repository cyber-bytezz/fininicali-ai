"""
Main runner script for the Financial Market Agent application.

This script sets up the Python path correctly and launches both 
the FastAPI orchestrator and Streamlit frontend.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

def main():
    """Run the FastAPI orchestrator and Streamlit frontend."""
    print("Starting Financial Market Agent Application...")
    
    # Create data directories if they don't exist
    os.makedirs(os.path.join(project_root, "data", "voice_output"), exist_ok=True)
    
    # Start the FastAPI orchestrator
    print("Starting FastAPI orchestrator on http://localhost:8000")
    orchestrator_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "orchestrator.app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=project_root
    )
    
    # Wait for the orchestrator to start
    print("Waiting for orchestrator to start...")
    time.sleep(3)
    
    # Start the Streamlit frontend
    print("Starting Streamlit frontend on http://localhost:8501")
    streamlit_process = subprocess.Popen(
        ["python", "-m", "streamlit", "run", "streamlit_app/app.py"],
        cwd=project_root
    )
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop processes on keyboard interrupt
        print("Stopping services...")
        orchestrator_process.terminate()
        streamlit_process.terminate()
        orchestrator_process.wait()
        streamlit_process.wait()
        print("Services stopped")

if __name__ == "__main__":
    main()
