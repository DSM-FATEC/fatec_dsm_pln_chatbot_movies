from pickle import load

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

MOVIES_DATASET = 'datasources/input/movie.csv'
MOVIES_CLEANED_LEMMA_DATASET = 'datasources/output/movies_cleaned.pickle'
TESTS_CLEANED_LEMMA_DATASET = 'datasources/output/tests_cleaned.pickle'
Y_DATASET = 'datasources/output/y.pickle'
NEGATIVE_SENTENCES = 'datasources/output/negative_sentences.pickle'
POSITIVE_SENTENCES = 'datasources/output/positive_sentences.pickle'
NEUTRAL_SENTENCES = 'datasources/output/neutral_sentences.pickle'


SENTIMENT_POSITIVE = 'positive'
SENTIMENT_NEUTRAL = 'neutral'
SENTIMENT_NEGATIVE = 'negative'


def token_is_valid(token):
    return not (
        token.like_num
        or token.is_punct
        or token.is_space
        or len(token) == 1
    )


def clean_text(sentence: str) -> str:
    nlp = spacy.load('pt_core_news_sm')

    doc = nlp(sentence.lower())
    tokens = []
    size = len(doc)

    for i, token in enumerate(doc):
        print(f'Cleaning token {i} of {size}')

        if token_is_valid(token):
            tokens.append(token.text)

    return ' '.join(tokens)


def preprocessing_lemma(sentence, nlp):
    tokens = [token.lemma_ for token in nlp(sentence)]
    tokens = ' '.join(tokens)

    return tokens


def get_sentiment(sentence: str) -> str:
    nlp = spacy.load('pt_core_news_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(sentence.lower())

    if doc._.polarity < -0.5:
        return SENTIMENT_NEGATIVE

    if doc._.polarity > 0.5:
        return SENTIMENT_POSITIVE

    return SENTIMENT_NEUTRAL


# def get_sentiment(sentence: str) -> str:
#     with open(Y_DATASET, 'rb') as r:
#         targets = load(r)

#     with open(MOVIES_CLEANED_LEMMA_DATASET, 'rb') as r:
#         all_sentences = load(r)

#     vectorizer = TfidfVectorizer()
#     tfidf = vectorizer.fit_transform(all_sentences)
#     model = DecisionTreeClassifier(criterion='entropy')
#     model.fit(tfidf, targets)

#     all_sentences.append(sentence)
#     dataset_tfidf = vectorizer.transform(all_sentences)

#     predictions = model.predict(dataset_tfidf)
#     score = predictions[-1]

#     if score == 0:
#         return SENTIMENT_NEGATIVE

#     return SENTIMENT_POSITIVE
