# Deployment Guide

This guide provides instructions for deploying the Multi-Agent Finance Assistant.

## Local Deployment

### Prerequisites

- Python 3.10+
- pip package manager
- (Optional) Docker and Docker Compose for containerized deployment

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd financial-market-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update API keys and configuration in `.env`

### Running the Application

1. Start the FastAPI backend:
   ```bash
   # Windows
   python -m orchestrator.app

   # Linux/macOS
   python -m orchestrator.app
   ```

2. Start the Streamlit frontend in a separate terminal:
   ```bash
   # Windows
   streamlit run streamlit_app/app.py

   # Linux/macOS
   streamlit run streamlit_app/app.py
   ```

3. Access the application:
   - FastAPI backend: http://localhost:8000
   - Streamlit frontend: http://localhost:8501

## Docker Deployment

### Prerequisites

- Docker and Docker Compose

### Deployment Steps

1. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update API keys and configuration in `.env`

2. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - FastAPI backend: http://localhost:8000
   - Streamlit frontend: http://localhost:8501

4. View logs:
   ```bash
   docker-compose logs -f
   ```

5. Stop the application:
   ```bash
   docker-compose down
   ```

## Cloud Deployment

### Streamlit Cloud

1. Create a Streamlit Cloud account at [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Configure the app:
   - Main file path: `streamlit_app/app.py`
   - Python version: 3.10
   - Add secrets (environment variables) in the Streamlit Cloud dashboard

### Heroku

1. Install the Heroku CLI
2. Log in to Heroku:
   ```bash
   heroku login
   ```

3. Create a new Heroku app:
   ```bash
   heroku create financial-market-agent
   ```

4. Add a Procfile to the repository:
   ```
   web: cd financial-market-agent && uvicorn orchestrator.app:app --host=0.0.0.0 --port=$PORT
   ```

5. Configure environment variables:
   ```bash
   heroku config:set TOGETHER_API_KEY=your_api_key
   heroku config:set ALPHAVANTAGE_API_KEY=your_api_key
   ```

6. Deploy the application:
   ```bash
   git push heroku main
   ```

## Troubleshooting

### Common Issues

1. **API Connection Issues**:
   - Check if the FastAPI service is running
   - Verify the API URL in the Streamlit app is correct
   - Check firewall settings if services are on different machines

2. **Missing Dependencies**:
   - Run `pip install -r requirements.txt` to ensure all dependencies are installed
   - For system dependencies, refer to the Dockerfile for required packages

3. **API Key Issues**:
   - Verify that API keys are correctly set in the `.env` file
   - For services like Together AI, ensure the API key is valid and has sufficient credits

4. **Voice Processing Issues**:
   - Ensure ffmpeg is installed for audio processing
   - Check that the required audio libraries are installed

### Getting Help

If you encounter issues not covered here, please:
1. Check the logs for detailed error messages
2. Search for similar issues in the repository's issue tracker
3. Open a new issue with detailed information about the problem
