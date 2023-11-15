from pickle import dump, load
from os import getenv, path
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from nltk import sent_tokenize, download

from extractors.movies_extractor import extract_movies
from parsers.spacy_parser import clean_text
from models.message_model import MessageModel
from responders.chatbot import get_answer_index


SENTENCES_PICKLE = 'datasources/sentences.pickle'
TOKENS_PICKLE = 'datasources/tokens.pickle'


if __name__ == '__main__':
    download('punkt')
    extract_movies()
