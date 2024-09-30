# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file for utilities
COPY ../../src/utils/requirements.txt .

# Install the required Python packages for utilities and helpers
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app source code into the container
COPY . .

# Set the entrypoint to the /src/utils directory
WORKDIR /app/src/utils

# Define the default command to be run (can be overwritten later)
CMD ["python", "logger.py"]
