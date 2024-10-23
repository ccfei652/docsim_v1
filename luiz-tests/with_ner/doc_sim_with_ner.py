import re
import pandas as pd
import numpy as np
from string import punctuation
import spacy
from spacy.lang.en import stop_words
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

ENTITY_WEIGHTS = {
    "ORG": 2.0,  # Organizações
    "PERSON": 1.0,  # Pessoas
    "GPE": 1.5,  # Localização
    "EVENT": 2.5  # Eventos
}


class WithNerDocumentProcessor:
    def __init__(self, model, nlp):
        self.model = model
        self.nlp = nlp
        self.stop_words = stop_words.STOP_WORDS
        self.punctuations = list(punctuation)

    def vectorize_text(self, text):
        words = self.preprocess(text)
        return self.document_vector(words)

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
        model_words = list(self.model.index_to_key)

        for word in sentence:
            if word in model_words:
                result_vector += self.model.get_vector(word)
            if word.lower() in model_words:
                result_vector += self.model.get_vector(word.lower())
            if word.title() in model_words:
                result_vector += self.model.get_vector(word.title())

        return result_vector

    @staticmethod
    def calculate_similarity(vector1, vector2):
        return cosine_similarity([vector1], [vector2]).flatten()[0]

    def extract_entities(self, text):
        preprocessed_text = self.preprocess(text)
        doc = self.nlp(" ".join(preprocessed_text))
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append((ent.text, ENTITY_WEIGHTS.get(ent.label_, 0.5)))

        return entities

    def extract_linked_entities(self, text):
        preprocessed_text = self.preprocess(text)
        doc = self.nlp(" ".join(preprocessed_text))
        linked_entities = {}

        for link_ent in doc._.linkedEntities:
            linked_entities[link_ent.get_id()] = {
                "description": link_ent.get_description(),
                "label": link_ent.get_label(),
                "original_alias": link_ent.original_alias,
                "super_entities": {
                    super_ent.get_id(): {
                        "description": super_ent.get_description(),
                        "label": super_ent.get_label(),
                        "original_alias": super_ent.original_alias,
                    } for super_ent in link_ent.get_super_entities()}
            }

        return linked_entities


class WithNerSimilarityAnalyzer:
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    @staticmethod
    def string_to_vector(string_vector):
        cleaned_string = re.sub(r'[\[\]\n]', '', string_vector)
        return np.fromstring(cleaned_string.strip(), sep=' ')

    def add_title_plus_summary_column(self):
        self.df['Title_Summary'] = self.df['Title'] + " " + self.df['Summary']

    def extract_and_add_entities_column(self):
        self.df['entities'] = self.df['Title_Summary'].apply(lambda x: self.processor.extract_entities(x))

    def extract_and_add_linked_entities_column(self):
        self.df['linked_entities'] = self.df['Title_Summary'].apply(lambda x: self.processor.extract_linked_entities(x))

    def apply_vectorization(self):
        self.df['title_vectors'] = self.df['Title'].apply(lambda x: self.processor.vectorize_text(x))
        self.df['summary_vector'] = self.df['Summary'].apply(lambda x: self.processor.vectorize_text(x))
        self.df['title_summary_vector'] = self.df['Title_Summary'].apply(lambda x: self.processor.vectorize_text(x))

    def calculate_similarities(self, user_input, user_entities, user_linked_entities):
        user_input_vector = self.processor.vectorize_text(user_input)
        self.df['title_similarity'] = self.df['title_vectors'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, x))
        self.df['summary_similarity'] = self.df['summary_vector'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, x))
        self.df['title_summary_similarity'] = self.df['title_summary_vector'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, x))

        self.df['entity_similarity'] = self.df['entities'].apply(
            lambda x: self.calculate_entity_similarity(user_entities, x))

        self.df['linked_entity_similarity'] = self.df['linked_entities'].apply(
            lambda x: self.calculate_linked_entity_similarity(user_linked_entities, x))

    @staticmethod
    def calculate_entity_similarity(user_entities, article_entities):
        score = 0
        for label, user_ent_list in user_entities.items():
            if label in article_entities:
                for user_ent, user_weight in user_ent_list:
                    for article_ent, article_weight in article_entities[label]:
                        if user_ent == article_ent:
                            score += user_weight + article_weight
        return score

    @staticmethod
    def calculate_linked_entity_similarity(user_linked_entities, article_linked_entities):
        score = 0
        for user_ent_id, user_ent_data in user_linked_entities.items():
            if user_ent_id in article_linked_entities:
                score += 1
            else:
                user_super_ents = user_ent_data['super_entities']
                for article_ent_id, article_ent_data in article_linked_entities.items():
                    article_super_ents = article_ent_data['super_entities']
                    for super_ent_id in user_super_ents:
                        if super_ent_id in article_super_ents:
                            score += 0.5
        return score

    def top_n_similar_documents(self, similarity_column, n=10):
        return self.df.sort_values(by=[similarity_column], ascending=[False]).head(n)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("entityLinker", last=True)

    word2vec_model = KeyedVectors.load('./word2vec-google-news-300.model')

    df = pd.read_csv('./titles_vectors.csv')

    # Inicializar processador e analisador
    processor = WithNerDocumentProcessor(model=word2vec_model, nlp=nlp)
    analyzer = WithNerSimilarityAnalyzer(df=df, processor=processor)

    # Adicionar a coluna de entidades
    # analyzer.add_title_plus_summary_column()
    # analyzer.extract_and_add_entities_column()
    # analyzer.extract_and_add_linked_entities_column()

    # Aplicar vetorização
    analyzer.apply_vectorization()

    # Salvar no CSV
    df.to_csv('titles_vectors.csv', index=False)

    # Definir input do usuário
    user_input = "how to predict user input inside a user interface"
    user_entities = processor.extract_entities(user_input)
    user_linked_entities = processor.extract_linked_entities(user_input)

    # Calcula similaridade
    analyzer.calculate_similarities(user_input, user_entities, user_linked_entities)

    # Exibir top 10 documentos semelhantes
    print("Top 10 documentos mais semelhantes por Title Similarity:")
    print(analyzer.top_n_similar_documents('title_similarity')[
              ['Title', 'title_similarity', 'entity_similarity', 'linked_entity_similarity']])

    print("\nTop 10 documentos mais semelhantes por Summary Similarity:")
    print(analyzer.top_n_similar_documents('summary_similarity')[
              ['Title', 'summary_similarity', 'entity_similarity', 'linked_entity_similarity']])

    print("\nTop 10 documentos mais semelhantes por Title + Summary Similarity:")
    print(analyzer.top_n_similar_documents('title_summary_similarity')[
              ['Title', 'title_summary_similarity', 'entity_similarity', 'linked_entity_similarity']])

    # Salvar no CSV
    df.to_csv('titles_vectors.csv', index=False)