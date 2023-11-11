FROM python:3.11.5-slim-bookworm

ENV PORT 8000

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download pt_core_news_sm

ENTRYPOINT ["python", "main.py"]
