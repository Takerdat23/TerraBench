#!/bin/bash

echo "Setting up docker image for python..."
docker build -t intercode-python -f docker/python.Dockerfile .
