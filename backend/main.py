from pickle import dump, load
from os import getenv, path
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from nltk import sent_tokenize, download

# from extractors.wikipedia_extractor import extract_cats
from parsers.spacy_parser import clean_text
from models.message_model import MessageModel
from responders.chatbot import get_answer_index
from extractors.movies_extractor import (
    Y_DATASET,
    MOVIES_CLEANED_LEMMA_DATASET,
    TESTS_CLEANED_LEMMA_DATASET,
)


SENTENCES_PICKLE = 'datasources/sentences.pickle'
TOKENS_PICKLE = 'datasources/tokens.pickle'

# Instanciando o FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_data_from_pickle():
    sentences = None
    tokens = None

    with open(Y_DATASET, 'rb') as r:
        y = load(r)

    with open(MOVIES_CLEANED_LEMMA_DATASET, 'rb') as r:
        x_movies_cleaned_lemma = load(r)

    with open(TESTS_CLEANED_LEMMA_DATASET, 'rb') as r:
        x_teste_cleaned_lemma = load(r)

    return y, x_movies_cleaned_lemma, x_teste_cleaned_lemma


@app.post('/msg')
def answer_msg(body: MessageModel) -> str:
    # Preprocessando mensagem
    preprocessed_message = clean_text(body.message)

    y, movies, teste = load_data_from_pickle()

    if body.message in ('oi', 'olá', 'salve', 'salve meu bom, suave?'):
        word_freq = Counter(movies)
        wordcloud_data = [word for word, _ in word_freq.items()]
        text = ', '.join(wordcloud_data[:1])

        return f'Olá! Como posso sanar suas dúvidas sobre filmes hoje? Algumas palavras que podem ser usadas: {text}'

    index = get_answer_index(preprocessed_sentences=movies,
                             preprocessed_message=preprocessed_message)
    if index is None:
        return "Desculpe, não consegui pensar em nenhuma resposta"

    try:
        return movies[index]
    except:
        return "Desculpe, não consegui pensar em nenhuma resposta"


if __name__ == '__main__':
    # Instala dependências do NLTK
    download('punkt')

    port = int(getenv('PORT', 8000))

    run(app='main:app', host='0.0.0.0', port=port)
