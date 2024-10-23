import requests

#### ConceptNet
def get_related_concepts(word):
    url = f"http://api.conceptnet.io/c/en/{word}"
    response = requests.get(url).json()
    related_concepts = [edge['end']['label'] for edge in response['edges']]
    return related_concepts

word = "cars"
# related_concepts = get_related_concepts(word)
# print(f"Conceitos relacionados a {word}: {related_concepts}")


#### Gensim
from gensim.models import KeyedVectors
word2vec_model = KeyedVectors.load('./word2vec-google-news-300.model')
def get_similar_words(word):
    similar_words = word2vec_model.most_similar(word)
    return similar_words

# Exemplo
similar_words = get_similar_words(word)
print(f"Palavras semelhantes a {word}: {similar_words}")


##### NLTK
from nltk.corpus import wordnet

# Função para buscar sinônimos
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

# Exemplo
synonyms = get_synonyms(word)
print(f"Sinônimos de {word}: {synonyms}")