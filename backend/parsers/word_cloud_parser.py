import re
from collections import Counter

def remove_html_tags(text: str) -> str:
    expression = re.compile(r'<[^>]+>|/>|br />|<br')
    cleaned__text = expression.sub('', text)

    return cleaned__text

def make_word_cloud(dataset):
    data = []

    for sentence in dataset:
        data.extend([str(word).lower() for word in sentence.split() if len(word) >= 5])

    counter = Counter(data)
    frequencies = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    wordcloud = [word for word, _ in frequencies]
    text = remove_html_tags(', '.join(wordcloud[:15]))

    return text
