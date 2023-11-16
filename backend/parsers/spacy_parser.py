import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


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


def is_sentence_negative(sentence: str, dataset, targets) -> bool:
    dataset = dataset.tolist()

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(dataset)
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(tfidf, targets)

    dataset.append(sentence)
    dataset_tfidf = vectorizer.transform(dataset)

    predictions = model.predict(dataset_tfidf)
    score = predictions[-1]

    return score == 0
