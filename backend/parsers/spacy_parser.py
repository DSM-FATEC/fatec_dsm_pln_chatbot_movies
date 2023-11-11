import spacy


def token_is_valid(token):
    return not (
        token.like_num
        or token.is_punct
        or token.is_space
        or len(token) == 1
    )


def clean_text(sentence):
    nlp = spacy.load('pt_core_news_sm')

    doc = nlp(sentence.lower())
    tokens = []
    size = len(doc)

    for i, token in enumerate(doc):
        print(f'Cleaning token {i} of {size}')

        if token_is_valid(token):
            tokens.append(token.text)

    return ' '.join(tokens)
