#!/bin/bash
docker build . -f ./deploy/Dockerfile -t pln-chatbot-api:v1.5 --platform linux/amd64

