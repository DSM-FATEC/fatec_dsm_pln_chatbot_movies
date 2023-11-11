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

    with open(SENTENCES_PICKLE, 'rb') as reader:
        sentences = load(reader)

    with open(TOKENS_PICKLE, 'rb') as reader:
        tokens = load(reader)

    return sentences, tokens


@app.post('/msg')
def answer_msg(body: MessageModel) -> str:
    # Preprocessando mensagem
    preprocessed_message = clean_text(body.message)

    sentences, tokens = load_data_from_pickle()

    if body.message in ('oi', 'olá', 'salve', 'salve meu bom, suave?'):
        word_freq = Counter(sentences)
        wordcloud_data = [word for word, _ in word_freq.items()]
        text = ', '.join(wordcloud_data[:1])

        return f'Olá! Como posso sanar suas dúvidas sobre gatos hoje? Algumas palavras que podem ser usadas: {text}'

    index = get_answer_index(preprocessed_sentences=sentences,
                             preprocessed_message=preprocessed_message)
    if index is None:
        return "Desculpe, não consegui pensar em nenhuma resposta"

    try:
        return tokens[index]
    except:
        return "Desculpe, não consegui pensar em nenhuma resposta"


if __name__ == '__main__':
    # Instala dependências do NLTK
    download('punkt')

    port = int(getenv('PORT', 8000))

    run(app='main:app', host='0.0.0.0', port=port)
