from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def get_answer_index(preprocessed_sentences, preprocessed_message, threshold = 0.25) -> str:
    # Adiciona o texto do usuário processado ao final das
    # sentenças limpas
    preprocessed_sentences.append(preprocessed_message)

    # Instancia vetorizador de TF-IDF
    vectorizer = TfidfVectorizer()

    # Vetoriza e transforma as sentenças preprocessadas
    vectorized_sentences = vectorizer.fit_transform(preprocessed_sentences)

    # Calcula a similaridade de conseno entre a ultima posição (pergunta usuário)
    # e as demais sentenças
    similarity = cosine_similarity(vectorized_sentences[-1], vectorized_sentences)

    # Obtém o indice da penúltima posição (maior corresopndencia)
    similarity_index = similarity.argsort()[0][-2]

    # Obtém o valor de similaridade
    similarity_score = similarity[0][similarity_index]

    if similarity_score < threshold:
        return None 

    # Acessa a lista de sentenças originais
    return similarity_index
