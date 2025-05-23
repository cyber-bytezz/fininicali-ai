"""
Streamlit entry point for cloud deployment.

This simplified version is designed to run on Streamlit Cloud without the FastAPI backend.
"""
import os
import sys
import json
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import the main streamlit app
from streamlit_app.app import main

# For Streamlit Cloud deployment
if __name__ == "__main__":
    main()
