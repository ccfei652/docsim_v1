import numpy as np

import spacy
from string import punctuation

from gensim.models import KeyedVectors
from spacy.lang.en import stop_words

from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
stop_words = stop_words.STOP_WORDS
punctuations = list(punctuation)

model = KeyedVectors.load('../word2vec-google-news-300.model')


def preprocess(sentence):
    sentence = nlp(sentence)
    # lemmatization
    sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]
    # removing stop words
    sentence = [word for word in sentence if word not in stop_words and word not in punctuations and word.isalpha()]
    return sentence


def document_vector(sentence: list[str]):
    result_vector = np.zeros(300)
    for word in sentence:
        try:
            result_vector += model.get_vector(word)
        except KeyError:
            if word.isnumeric():
                word = "0" * len(word)
                result_vector += model.get_vector(word)
            else:
                result_vector += model.get_vector("unknown")
    return result_vector


def vector_matrix(sentences):
    x = len(sentences)
    y = 300
    matrix = np.zeros((x, y))

    for i, sentence in enumerate(sentences):
        sentence_preprocessed = preprocess(sentence)
        matrix[i] = document_vector(sentence_preprocessed)

    return matrix


def similar_documents(user_input, matrix, original_docs: list[str]):
    input_preprocess = preprocess(user_input)
    input_vector = document_vector(input_preprocess)

    similarities = cosine_similarity([input_vector], matrix).flatten()
    docs_similarity_sorted_indexes = np.argsort(similarities)[-10:][::-1]
    sorted_similarity = similarities[docs_similarity_sorted_indexes]

    most_related_docs = [original_docs[i] for i in docs_similarity_sorted_indexes]

    return list(zip(most_related_docs, sorted_similarity))


def calculate_similarity_single(input_vector, doc_vector):
    return cosine_similarity([input_vector], [doc_vector]).flatten()[0]
