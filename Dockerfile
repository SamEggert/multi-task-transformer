# Use an official Python 3.12 image as a base
FROM python:3.12-slim

# Set a working directory
WORKDIR /app

# Install system dependencies (for torch, transformers, etc.)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your code into the container
COPY . .

# Set the default command to run your training script
CMD ["python", "trainMultiTaskEncoder.py"] 