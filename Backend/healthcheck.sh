#!/bin/bash

# Check if Ollama is accessible
if ! curl -sf http://host.docker.internal:11434 > /dev/null; then
    echo "Error: Cannot connect to Ollama service"
    exit 1
fi

# Check if our service is running
if ! curl -sf http://localhost:8000 > /dev/null; then
    echo "Error: FastAPI service is not responding"
    exit 1
fi

echo "All services are healthy"
exit 0 