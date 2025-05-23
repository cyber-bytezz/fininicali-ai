"""
Streamlit Cloud entry point.

This file serves as the main entry point for Streamlit Cloud deployment.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import the main app function
from streamlit_app.app import main

# Run the app in cloud mode
if __name__ == "__main__":
    main(cloud_mode=True)
