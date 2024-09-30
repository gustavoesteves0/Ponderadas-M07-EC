# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file for models
COPY ../../src/models/requirements.txt .

# Install the required Python packages for model training and evaluation
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app source code into the container
COPY . .

# Set the entrypoint to the /src/models directory
WORKDIR /app/src/models

# Expose any necessary ports (if applicable)
# EXPOSE 8000

# Define the default command to be run (can be overwritten later)
CMD ["python", "train.py"]
