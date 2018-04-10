"""The baseline is one-hot encoded word vectors."""
import numpy as np

from reach import Reach
from sklearn.feature_extraction.text import CountVectorizer


def baseline(text, keep_n=10000):
    """Create a one-hot encoded baseline vector space."""
    c = CountVectorizer(text, max_features=keep_n)
    c.fit(text)

    words = c.get_feature_names()
    words = ["UNK"] + words
    vectors = np.eye(len(words))
    return Reach(vectors, list(words), unk_index=0)
