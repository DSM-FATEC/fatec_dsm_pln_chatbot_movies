import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


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

    print(f'Classificação: {score}')

    return score == 0
