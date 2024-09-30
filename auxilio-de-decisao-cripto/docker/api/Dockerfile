# Use an official Fedora image as the base
FROM fedora:38

# Set environment variables to prevent Python from writing .pyc files and to buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install Python 3.11, pip, and build dependencies
RUN dnf update -y && \
    dnf install -y python3.11 python3.11-devel python3.11-pip gcc postgresql-devel && \
    dnf clean all

# Create symlinks for python3 and pip3 to point to Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip3

# Upgrade pip, setuptools, and wheel
RUN pip3 install --upgrade pip setuptools wheel

# Copy the application requirements file to the container
COPY ../../src/api/requirements.txt .

# Install any necessary dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Force reinstall NumPy and Pandas to ensure compatibility
RUN pip3 install --upgrade --force-reinstall numpy==1.26.4 pandas==2.2.3

# Remove build dependencies to reduce image size
RUN dnf remove -y gcc postgresql-devel python3.11-devel && \
    dnf clean all

# Clean Python cache to prevent binary incompatibility issues
RUN find . -type d -name "__pycache__" -exec rm -r {} + && \
    find . -name "*.pyc" -delete

# Copy the entire app directory into the container
COPY . .

# Expose the port your API will run on
EXPOSE 8000

# Command to run the API server (FastAPI example)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
