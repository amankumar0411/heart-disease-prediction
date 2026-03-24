# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements from the same folder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code into the container
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=backend_api.py
ENV PYTHONUNBUFFERED=1

# Start Flask app
CMD ["python", "backend_api.py"]
