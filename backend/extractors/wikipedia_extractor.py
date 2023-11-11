from goose3 import Goose


CATS_ARTICLE_URL = 'https://en.wikipedia.org/wiki/Cat'


def extract_cats():
    goose = Goose()
    article = goose.extract(CATS_ARTICLE_URL)

    return article.cleaned_text
