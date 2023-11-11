from pickle import dump, load
from os import getenv, path
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from nltk import sent_tokenize, download

from extractors.wikipedia_extractor import extract_cats
from parsers.spacy_parser import clean_text
from models.message_model import MessageModel
from responders.chatbot import get_answer_index


SENTENCES_PICKLE = 'datasources/sentences.pickle'
TOKENS_PICKLE = 'datasources/tokens.pickle'


def extract_data_to_pickle() -> None:
    cats_article = extract_cats()

    # Criando tokens NLTK do texto bruto
    cats_article_tokens = sent_tokenize(cats_article)

    # Preprocessando sentenças e limpando texto bruto
    preprocessed_sentences = [clean_text(token) for token in cats_article_tokens]

    with open(SENTENCES_PICKLE, 'wb') as writer:
        dump(preprocessed_sentences, writer)

    with open(TOKENS_PICKLE, 'wb') as writer:
        dump(cats_article_tokens, writer)


if __name__ == '__main__':
    download('punkt')
    extract_data_to_pickle()
