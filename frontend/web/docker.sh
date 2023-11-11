#!/bin/bash

versao="1.3"

# Compila imagem docker
docker build . -f ./deploy/Dockerfile -t "pln-chatbot-front:v$versao" --platform linux/amd64

# Tagueia a imagem docker para o registry da GCP
docker tag "pln-chatbot-front:v$versao" "southamerica-east1-docker.pkg.dev/pessoal-374013/dockers/pln-chatbot-front:v$versao"

# Envia imagem para a GCP
docker push "southamerica-east1-docker.pkg.dev/pessoal-374013/dockers/pln-chatbot-front:v$versao"

