# Ubuntu focal as base image
FROM ubuntu:20.04

# Install required dependencies
RUN apt-get update && apt-get install -y gcc wget make

# Copy the main application to the container
COPY llama2.c /app/llama2.c

# Set the working directory
WORKDIR /app/llama2.c

# Run as a long-lived container
CMD ["tail", "-f", "/dev/null"]

    