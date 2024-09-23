import pandas as pd

from doc_treatment import vector_matrix, similar_documents

df = pd.read_csv('../arXiv-DataFrame.csv')

summaries = df['Summary'].to_list()

matrix = vector_matrix(summaries)

matrix_df = pd.DataFrame(matrix, index=summaries)
df.to_csv('summary_vectors.csv')

user_input = "autonomous cars"

similar_docs = similar_documents(user_input, matrix, summaries)

for i, (titulo, similaridade) in enumerate(similar_docs, 1):
    print(f"{i}. {titulo} - Similaridade: {similaridade:.4f}")