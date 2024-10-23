import gensim.downloader
from gensim.models import KeyedVectors

model = gensim.downloader.load('word2vec-google-news-300')
model_save_path = './luiz-tests/with_ner/word2vec-google-news-300.model'
model.save(model_save_path)

model = KeyedVectors.load('word2vec-google-news-300.model')
