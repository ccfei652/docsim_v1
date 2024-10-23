import re
import pandas as pd
import numpy as np
from string import punctuation
import spacy
from spacy.lang.en import stop_words
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


class WithUnknownDocumentProcessor:
    def __init__(self, model, nlp):
        self.model = model
        self.nlp = nlp
        self.stop_words = stop_words.STOP_WORDS
        self.punctuations = list(punctuation)

    def preprocess(self, text):
        sentence = self.nlp(text)
        # lemmatization
        sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]
        # removing stop words
        sentence = [
            word for word in sentence
            if word not in self.stop_words
               and word not in self.punctuations and word.isalpha()
        ]
        return sentence

    def document_vector(self, sentence: list[str]):
        result_vector = np.zeros(300)
        for word in sentence:
            try:
                result_vector += self.model.get_vector(word)
            except KeyError:
                if word.isnumeric():
                    word = "0" * len(word)
                    result_vector += self.model.get_vector(word)
                else:
                    result_vector += self.model.get_vector("unknown")
        return result_vector

    def vector_matrix(self, sentences):
        x = len(sentences)
        y = 300
        matrix = np.zeros((x, y))

        for i, sentence in enumerate(sentences):
            sentence_preprocessed = self.preprocess(sentence)
            matrix[i] = self.document_vector(sentence_preprocessed)

        return matrix

    @staticmethod
    def calculate_similarity(vector1, vector2):
        return cosine_similarity([vector1], [vector2]).flatten()[0]

    def vectorize_text(self, text):
        words = self.preprocess(text)
        return self.document_vector(words)


class WithUnknownSimilarityAnalyzer:
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    @staticmethod
    def string_to_vector(string_vector):
        cleaned_string = re.sub(r'[\[\]\n]', '', string_vector)
        return np.fromstring(cleaned_string.strip(), sep=' ')

    def apply_vectorization(self):
        self.df['Title_Summary'] = self.df['Title'] + " " + self.df['Summary']
        self.df['title_vectors'] = self.df['Title'].apply(lambda x: self.processor.vectorize_text(x))
        self.df['summary_vector'] = self.df['Summary'].apply(lambda x: self.processor.vectorize_text(x))
        self.df['title_summary_vector'] = self.df['Title_Summary'].apply(lambda x: self.processor.vectorize_text(x))

    def calculate_similarities(self, user_input):
        user_input_vector = self.processor.vectorize_text(user_input)
        self.df['title_similarity'] = self.df['title_vectors'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, x))
        self.df['summary_similarity'] = self.df['summary_vector'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, x))
        self.df['title_summary_similarity'] = self.df['title_summary_vector'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, x))

    def top_n_similar_documents(self, similarity_column, n=10):
        return self.df.nlargest(n, similarity_column)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    word2vec_model = KeyedVectors.load('./word2vec-google-news-300.model')

    df = pd.read_csv('./arXiv-DataFrame.csv')

    # Inicializar processador e analisador
    processor = WithUnknownDocumentProcessor(model=word2vec_model, nlp=nlp)
    analyzer = WithUnknownSimilarityAnalyzer(df=df, processor=processor)

    # Aplicar vetorização
    analyzer.apply_vectorization()

    # Definir input do usuário
    user_input = "autonomous cars"
    analyzer.calculate_similarities(user_input)

    # Exibir top 10 documentos semelhantes
    print("Top 10 documentos mais semelhantes por Title Similarity:")
    print(analyzer.top_n_similar_documents('title_similarity')[['Title', 'title_similarity']])

    print("\nTop 10 documentos mais semelhantes por Summary Similarity:")
    print(analyzer.top_n_similar_documents('summary_similarity')[['Title', 'summary_similarity']])

    print("\nTop 10 documentos mais semelhantes por Title + Summary Similarity:")
    print(analyzer.top_n_similar_documents('title_summary_similarity')[['Title', 'title_summary_similarity']])

    # Salvar no CSV
    df.to_csv('titles_vectors.csv', index=False)
