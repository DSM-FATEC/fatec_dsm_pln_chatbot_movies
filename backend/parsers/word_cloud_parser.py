import re
from collections import Counter

def remove_html_tags(text: str) -> str:
    expression = re.compile(r'<[^>]+>|/>|br />')
    cleaned__text = expression.sub('', text)

    return cleaned__text

def make_word_cloud(dataset):
    word_freq = Counter(dataset)
    wordcloud_data = [word for word, _ in word_freq.items()]
    words = str(wordcloud_data[0]).split()[:15]
    text = remove_html_tags(', '.join(words))

    return text
