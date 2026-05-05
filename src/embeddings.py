from gensim.models import Word2Vec
import numpy as np

def train_word2vec(X_train):
    sentences = [text.split() for text in X_train]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
    return model

def vectorize_text(model, X):
    def avg_vector(words):
        vec = np.zeros(100)
        count = 0
        for word in words:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        return vec / count if count > 0 else vec
    return np.array([avg_vector(text.split()) for text in X])