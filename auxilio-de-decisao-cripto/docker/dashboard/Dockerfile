# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application requirements file to the container
COPY ../../src/dashboard/requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app source code into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set Streamlit to use the host's IP to make the dashboard accessible
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the PYTHONPATH to make sure /app/src is on the path
ENV PYTHONPATH="/app:/app/src"

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/dashboard/app.py"]
