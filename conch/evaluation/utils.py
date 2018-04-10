"""Utility functions."""
import regex as re
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

# Regex used to convert BIO sequences to indices.
BIO_FINDER = re.compile(r"BI*")


def bio_to_index(bio_tags):
    """Convert a list of lists of bio_tags to indices of chunks."""
    tags = []

    for idx, bio_sent in enumerate(bio_tags):
        no_labels = "".join([x.split("-")[0] for x in bio_sent])
        tags.append([(x.start(), x.end(), bio_sent[x.start()].split("-")[1])
                     for x in BIO_FINDER.finditer(no_labels)])

    return tags


def evaluate_k(true, pred, average='micro'):
    """
    Evaluate a predicted vector of k nearest neighbors for each value of k.

    Parameters
    ==========
    true : list of string
        The true labels
    pred : lists of lists
        Each sublist contains k strings, which are the k nearest neighbors,
        ordered by their similarity.
    average : string or None, optional, default 'weighted'
        The averaging to use in the scoring function.

    Returns
    =======
    scores : list of lists
        An F-score for each class for each value of k.

    """
    s = []
    for x in range(1, len(pred[0])):
        p = [Counter(p[:x]).most_common(1)[0][0] for p in pred]
        score = precision_recall_fscore_support(true, p, average=average)
        score = np.array(score).tolist()
        s.append(score)

    return s


def to_conll(pred, gold, outputpath):
    """Convert pred and gold BIO sequences to .conll format."""
    assert(len(pred) == len(gold))

    with open(outputpath, 'w') as f:
        for x, y in zip(gold, pred):
            f.write("_ _ {0} {1}\n".format(x, y))
