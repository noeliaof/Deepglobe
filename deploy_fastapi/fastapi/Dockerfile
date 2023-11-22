#FROM tiangolo/uvicorn-gunicorn:python3.11
FROM python:3.11-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx \
                       libglib2.0-0 && \
    apt-get autoclean

# Set the working directory
WORKDIR /fastapi

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt /fastapi/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY models/final_model.pth /fastapi/models/final_model.pth
COPY data/*.csv /fastapi/data/
COPY . /fastapi/

# Expose the port
EXPOSE 8000

# Define the command to run on container start
CMD ["python", "api.py"]
