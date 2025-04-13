# Use an official Python base image
FROM python:3.12.7-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     portaudio19-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    portaudio19-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Copy all relevant files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure .env is read at runtime (handled in app.py using dotenv or os.environ)
# If using python-dotenv, make sure it's in requirements.txt

# Run the app
CMD ["python", "web_second_interface.py"]
