#!/bin/bash

# Define your image name and tag
IMAGE_NAME="jmendoza9/llama2-app"
TAG="latest"

# Build the Docker image
docker build -t "$IMAGE_NAME" .

# Tag the Docker image
docker tag "$IMAGE_NAME" "$IMAGE_NAME:$TAG"

# Push the Docker image to the registry
docker push "$IMAGE_NAME:$TAG"

