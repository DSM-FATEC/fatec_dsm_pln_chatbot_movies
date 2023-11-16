from pickle import load
from os import getenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from nltk import download

# from extractors.wikipedia_extractor import extract_cats
from parsers.spacy_parser import (
    clean_text,
    get_sentiment,
    is_sentence_negative,
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_POSITIVE
)
from parsers.word_cloud_parser import make_word_cloud, remove_html_tags
from models.message_model import MessageModel, MessageModelResponse
from responders.chatbot import get_answer_index
from extractors.movies_extractor import (
    Y_DATASET,
    MOVIES_CLEANED_LEMMA_DATASET,
    TESTS_CLEANED_LEMMA_DATASET,
    NEGATIVE_SENTENCES,
    POSITIVE_SENTENCES,
    NEUTRAL_SENTENCES,
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
def answer_msg(body: MessageModel) -> MessageModelResponse:
    question = clean_text(body.message)

    with open(Y_DATASET, 'rb') as r:
        targets = load(r)

    with open(MOVIES_CLEANED_LEMMA_DATASET, 'rb') as r:
        all_sentences = load(r)

    with open(TESTS_CLEANED_LEMMA_DATASET, 'rb') as r:
        tests = load(r)

    with open(NEGATIVE_SENTENCES, 'rb') as r:
        negative_sentences = load(r)

    with open(POSITIVE_SENTENCES, 'rb') as r:
        positive_sentences = load(r)

    with open(NEUTRAL_SENTENCES, 'rb') as r:
        neutral_sentences = load(r)

    if body.message in ('hi', 'hello'):
        return MessageModelResponse(answer=f'{HELLO_MESSAGE} {make_word_cloud(all_sentences)}',
                                    sentiment=SENTIMENT_NEUTRAL)

    # is_question_negative = is_sentence_negative(question, tests[:len(targets)], targets)
    question_sentiment = get_sentiment(question)

    answer_sentences = neutral_sentences

    if question_sentiment == SENTIMENT_NEGATIVE:
        answer_sentences = negative_sentences
    elif question_sentiment == SENTIMENT_POSITIVE:
        answer_sentences = positive_sentences

    index = get_answer_index(preprocessed_sentences=answer_sentences,
                             preprocessed_message=question)

    answer = SORRY_MESSAGE
    try:
        answer = remove_html_tags(answer_sentences[index])
    except:
        answer = SORRY_MESSAGE

    return MessageModelResponse(answer=answer,
                                sentiment=question_sentiment)


if __name__ == '__main__':
    # Instala dependências do NLTK
    download('punkt')

    port = int(getenv('PORT', 8000))

    run(app='main:app', host='0.0.0.0', port=port, reload=True)
