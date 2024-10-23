import re

import pandas as pd
import numpy as np
from doc_treatment import calculate_similarity_single, preprocess, document_vector

df = pd.read_csv('luiz-tests/with_unkown/titles_vectors.csv')

# df['Title_Summary'] = df['Title'] + " " + df['Summary']
#
# df['title_vectors'] = df['Title'].apply(lambda title: document_vector(preprocess(title)))
#
# df['summary_vector'] = df['Summary'].apply(lambda summary: document_vector(preprocess(summary)))
#
# df['title_summary_vector'] = df['Title_Summary'].apply(lambda title_summary: document_vector(preprocess(title_summary)))
#
# df.to_csv('titles_vectors.csv', index=False)

print(df.head())


def string_to_vector(string_vector):
    cleaned_string = re.sub(r'[\[\]\n]', '', string_vector)

    vector = np.fromstring(cleaned_string.strip(), sep=' ')

    return vector

#
# df['title_vectors'] = df['title_vectors'].apply(string_to_vector)
# df['summary_vector'] = df['summary_vector'].apply(string_to_vector)
# df['title_summary_vector'] = df['title_summary_vector'].apply(string_to_vector)
#
# user_input = "autonomous cars"
# user_input_vector = document_vector(preprocess(user_input))
#
# df['title_similarity'] = df['title_vectors'].apply(
#     lambda x: calculate_similarity_single(user_input_vector, np.array(x)))
# df['summary_similarity'] = df['summary_vector'].apply(
#     lambda x: calculate_similarity_single(user_input_vector, np.array(x)))
# df['title_summary_similarity'] = df['title_summary_vector'].apply(
#     lambda x: calculate_similarity_single(user_input_vector, np.array(x)))

top_10_title_similarity = df.nlargest(10, 'title_similarity')
print("Top 10 documentos mais semelhantes por Title Similarity:")
print(top_10_title_similarity[['Title', 'title_similarity']].to_string())

top_10_summary_similarity = df.nlargest(10, 'summary_similarity')
print("\nTop 10 documentos mais semelhantes por Summary Similarity:")
print(top_10_summary_similarity[['Title', 'summary_similarity']].to_string())

top_10_title_summary_similarity = df.nlargest(10, 'title_summary_similarity')
print("\nTop 10 documentos mais semelhantes por Title + Summary Similarity:")
print(top_10_title_summary_similarity[['Title', 'title_summary_similarity']].to_string())

df.to_csv('titles_vectors.csv', index=False)