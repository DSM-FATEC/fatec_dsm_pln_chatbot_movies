import re
from pickle import dump

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split

from parsers.spacy_parser import (
    get_sentiment,
    SENTIMENT_NEGATIVE,
    SENTIMENT_POSITIVE,
    SENTIMENT_NEUTRAL
)


MOVIES_DATASET = 'datasources/input/movie.csv'
MOVIES_CLEANED_LEMMA_DATASET = 'datasources/output/movies_cleaned.pickle'
TESTS_CLEANED_LEMMA_DATASET = 'datasources/output/tests_cleaned.pickle'
Y_DATASET = 'datasources/output/y.pickle'
NEGATIVE_SENTENCES = 'datasources/output/negative_sentences.pickle'
POSITIVE_SENTENCES = 'datasources/output/positive_sentences.pickle'
NEUTRAL_SENTENCES = 'datasources/output/neutral_sentences.pickle'


def preprocessing(sentence, nlp):
    sentence = re.sub(r'@[A-Za-z0-9]+', ' ', str(sentence))
    sentence = sentence.lower()
    tokens = [token.text for token in nlp(sentence) if not (token.is_stop
                                                            or token.like_num
                                                            or token.is_punct
                                                            or token.is_space
                                                            or len(token) == 1)]
    tokens = ' '.join(tokens)

    return tokens


def preprocessing_lemma(sentence, nlp):
    tokens = [token.lemma_ for token in nlp(sentence)]
    tokens = ' '.join(tokens)

    return tokens


def extract_movies():
    nlp = spacy.load('en_core_web_sm')
    df = pd.read_csv(MOVIES_DATASET,
                     header=0,

                     skiprows=1,
                     names=['Texto', 'Classificacao'],
                     encoding='latin1')

    print('Dataset loaded')

    x = df.iloc[:,0].values
    y = df.iloc[:,1].values
    x,_, y,_ = train_test_split(x, y)

    print('Train/Test data splitted')

    x_movies_cleaned = [preprocessing(movies, nlp) for movies in x]

    print('Train/Test data cleaned')

    x_movies_cleaned_lemma = [preprocessing_lemma(movies, nlp) for movies in x_movies_cleaned]

    print('Train/Test data lemma processed')

    negative_sentences = []
    positive_sentences = []
    neutral_sentences = []

    for sentence in x.tolist():
        sentence_sentiment = get_sentiment(sentence)

        if sentence_sentiment == SENTIMENT_NEGATIVE:
            negative_sentences.append(sentence)
        elif sentence_sentiment == SENTIMENT_POSITIVE:
            positive_sentences.append(sentence)
        elif sentence_sentiment == SENTIMENT_NEUTRAL:
            neutral_sentences.append(sentence)

    print('Sentiment data processed')

    with open(Y_DATASET, 'wb') as w:
        dump(y, w)

    with open(MOVIES_CLEANED_LEMMA_DATASET, 'wb') as w:
        dump(x.tolist(), w)

    with open(TESTS_CLEANED_LEMMA_DATASET, 'wb') as w:
        dump(x_movies_cleaned_lemma, w)

    print('Train/Test data saved')

    with open(NEGATIVE_SENTENCES, 'wb') as w:
        dump(negative_sentences, w)

    with open(POSITIVE_SENTENCES, 'wb') as w:
        dump(positive_sentences, w)

    with open(NEUTRAL_SENTENCES, 'wb') as w:
        dump(neutral_sentences, w)

    print('Sentiment data saved')
