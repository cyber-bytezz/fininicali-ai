@echo off
echo Starting Financial Market Agent Application...

REM Create necessary data directories
mkdir data 2>nul
mkdir data\voice_output 2>nul

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please update the .env file with your API keys before continuing.
    pause
)

REM Activate virtual environment
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Set PYTHONPATH to include the project root
set PYTHONPATH=%CD%

REM Run the application
echo Starting services...
python run_app.py

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat
