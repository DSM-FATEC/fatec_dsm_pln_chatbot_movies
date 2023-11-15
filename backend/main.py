from pickle import load
from os import getenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from nltk import download

# from extractors.wikipedia_extractor import extract_cats
from parsers.spacy_parser import clean_text, is_sentence_negative
from parsers.word_cloud_parser import make_word_cloud, remove_html_tags
from models.message_model import MessageModel
from responders.chatbot import get_answer_index
from extractors.movies_extractor import (
    Y_DATASET,
    MOVIES_CLEANED_LEMMA_DATASET,
    TESTS_CLEANED_LEMMA_DATASET,
)


# SENTENCES_PICKLE = 'datasources/sentences.pickle'
# TOKENS_PICKLE = 'datasources/tokens.pickle'

HELLO_MESSAGE = "Hello! let's talk about movies? Some keywords that might be used:"
SORRY_MESSAGE = "Sorry, i couldn't find any answer to your question, try again with other words"

# Instanciando o FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/msg')
def answer_msg(body: MessageModel) -> str:
    preprocessed_message = clean_text(body.message)

    with open(Y_DATASET, 'rb') as r:
        targets = load(r)

    with open(MOVIES_CLEANED_LEMMA_DATASET, 'rb') as r:
        movies = load(r)

    with open(TESTS_CLEANED_LEMMA_DATASET, 'rb') as r:
        tests = load(r)

    if body.message in ('hi', 'hello'):
        return f'{HELLO_MESSAGE} {make_word_cloud(movies)}'

    index = get_answer_index(preprocessed_sentences=tests,
                             preprocessed_message=preprocessed_message)

    print(is_sentence_negative(preprocessed_message, movies, targets))

    try:
        return remove_html_tags(movies[index])
    except:
        return SORRY_MESSAGE


if __name__ == '__main__':
    # Instala dependências do NLTK
    download('punkt')

    port = int(getenv('PORT', 8000))

    run(app='main:app', host='0.0.0.0', port=port, reload=True)
