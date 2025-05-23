FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    libasound-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/voice_output

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV STREAMLIT_PORT=8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
python -m orchestrator.app &\n\
streamlit run streamlit_app/app.py\n'\
> /app/start.sh && chmod +x /app/start.sh

# Run both services
CMD ["/app/start.sh"]
