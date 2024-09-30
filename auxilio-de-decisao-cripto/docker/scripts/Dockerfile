# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file for scripts
COPY ./../src/scripts/requirements.txt .

# Install the required Python packages for data processing and utilities
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app source code into the container
COPY . .

# Set the entrypoint to the /src/scripts directory
WORKDIR /app/src/scripts

# Define the default command to be run (can be overwritten later)
CMD ["python3", "data_cleaning.py"]
