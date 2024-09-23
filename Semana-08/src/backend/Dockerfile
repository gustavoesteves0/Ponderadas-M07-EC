# Use an official Python image as a base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt first to cache dependencies
COPY requirements.txt .

ENV PIP_DEFAULT_TIMEOUT=2000

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to start the backend service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]