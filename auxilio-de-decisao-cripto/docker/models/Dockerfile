# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements file into the container
COPY ../../src/models/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/src/models


# Define the entry point or command to run your models
# This can be adjusted based on how you want to use your models (e.g., for training, API server, etc.)
CMD ["python", "main.py"]
