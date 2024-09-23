import pandas as pd

from doc_treatment import vector_matrix, similar_documents

df = pd.read_csv('../arXiv-DataFrame.csv')

df['Title_Summary'] = df['Title'] + ' ' + df['Summary']

titles = df['Title'].to_list()

title_summaries = df['Title_Summary'].to_list()

matrix = vector_matrix(title_summaries)

matrix_df = pd.DataFrame(matrix, index=title_summaries)
df.to_csv('title_summaries_vectors.csv')

user_input = "autonomous cars"

similar_docs = similar_documents(user_input, matrix, title_summaries)

for i, (titulo, similaridade) in enumerate(similar_docs, 1):
    print(f"{i}. {titles[i]} - Similaridade: {similaridade:.4f}")