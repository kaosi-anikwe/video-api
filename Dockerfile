# Use the official Ubuntu image
FROM ubuntu:latest

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y

# Install required dependencies
RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    software-properties-common \
    ffmpeg \
    git

# Add deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv

# Clone the repository
RUN git clone https://github.com/kaosi-anikwe/video-api.git /root/video-api
WORKDIR /root/video-api

# Create and activate a virtual environment
RUN python3.10 -m venv env
RUN . env/bin/activate

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements/requirements.txt
RUN pip install -r requirements/pt2.txt
RUN pip install .
RUN pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata

# Download model checkpoint
RUN mkdir checkpoints
RUN wget -O checkpoints/svd_xt.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true

# Set environment variables
ENV VIDEO_DIR=/root/video-api/outputs/video
ENV THUMBNAIL_DIR=/root/video-api/outputs/thumbnail
ENV SECRET_KEY=niftyverse
ENV SECURITY_PASSWORD_SALT=niftyverse

# Create log directory
RUN mkdir log

# Start the application
CMD ["sh", "-c", "nohup gunicorn -w 4 -b :5000 run:app >> log/run.log 2>&1 & && nohup python -u worker.py >> log/run.log 2>&1 &"]
