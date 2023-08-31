import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Define theme-specific keywords
themes = {
    'sports': ['soccer', 'basketball', 'tennis'],
    'technology': ['computer', 'smartphone', 'internet'],
    'food': ['pizza', 'hamburger', 'pasta'],
    'nothing' : ['genral', 'text', 'nothing', 'place', 'holder']
}

# Calculate theme vectors
theme_vectors = {}
for theme, keywords in themes.items():
    keyword_vectors = model.encode(keywords)
    theme_vectors[theme] = np.mean(keyword_vectors, axis=0)

# Sample sentences from field notes
sentences = [
    'The soccer match was intense.',
    'She bought a new smartphone.',
    'We had delicious pasta for dinner.',
    'nothing meaningful on this sentence.'
]

# Annotate sentences
for sentence in sentences:
    # Convert sentence to a vector
    sentence_vector = model.encode([sentence])[0]

    # Calculate similarity between sentence vector and theme vectors
    similarities = {}
    for theme, theme_vector in theme_vectors.items():
        similarity = cosine_similarity([sentence_vector], [theme_vector])[0][0]
        similarities[theme] = similarity

    # Assign the most relevant theme
    most_relevant_theme = max(similarities, key=similarities.get)
    print(f'Sentence: "{sentence}" - Annotated theme: {most_relevant_theme}')

