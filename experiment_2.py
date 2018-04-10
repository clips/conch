"""
Run transfer experiments.

Running this file will replicate experiment 2 from the paper.
"""
import numpy as np
import json

from conch.evaluation.intrinsic import evaluate_transfer
from reach import Reach
from conch.conch import compose, reciprocal
from conch.evaluation.utils import evaluate_k
from conch.preprocessing.baseline import baseline
from itertools import chain


def experiment(parsed_train,
               gold_chunks_train,
               parsed_test,
               gold_chunks_test,
               f1,
               f2,
               embeddings,
               context_function,
               window,
               k,
               use_focus=True):
    """Run an experiment with transfer evaluation."""
    _, np_chunks_train = zip(*parsed_train)
    _, np_chunks_test = zip(*parsed_test)

    phrase_embeddings_train = compose(parsed_train,
                                      f1=f1,
                                      f2=f2,
                                      window=window,
                                      embeddings=embeddings,
                                      context_function=context_function,
                                      use_focus=use_focus)

    phrase_embeddings_test = compose(parsed_test,
                                     f1=f1,
                                     f2=f2,
                                     window=window,
                                     embeddings=embeddings,
                                     context_function=context_function,
                                     use_focus=use_focus)

    result = evaluate_transfer(gold_chunks_train,
                               np_chunks_train,
                               gold_chunks_test,
                               np_chunks_test,
                               phrase_embeddings_train,
                               phrase_embeddings_test,
                               k=k)

    return result


if __name__ == "__main__":

    scores = {}
    parsed_train = json.load(open("data/partners_uima.json"))
    parsed_train = list(zip(*sorted(parsed_train.items())))[1]

    gold_train = json.load(open("data/partners_gold.json"))
    gold_train = list(zip(*sorted(gold_train.items())))[1]

    parsed_test = json.load(open("data/beth_uima.json"))
    parsed_test = list(zip(*sorted(parsed_test.items())))[1]

    gold_test = json.load(open("data/beth_gold.json"))
    gold_test = list(zip(*sorted(gold_test.items())))[1]

    txt, gold_chunks_train = zip(*gold_train)
    _, gold_chunks_test = zip(*gold_test)

    embeddings = Reach.load("")

    for a, b in zip(parsed_train, gold_train):
        assert len(a[0]) == len(b[0])

    for a, b in zip(parsed_test, gold_test):
        assert len(a[0]) == len(b[0])

    knn_focus = experiment(parsed_train,
                           gold_chunks_train,
                           parsed_test,
                           gold_chunks_test,
                           np.mean,
                           np.mean,
                           embeddings,
                           reciprocal,
                           0,
                           k=1,
                           use_focus=True)

    knn_full = experiment(parsed_train,
                          gold_chunks_train,
                          parsed_test,
                          gold_chunks_test,
                          np.mean,
                          np.mean,
                          embeddings,
                          reciprocal,
                          10,
                          k=1,
                          use_focus=True)

    knn_context = experiment(parsed_train,
                             gold_chunks_train,
                             parsed_test,
                             gold_chunks_test,
                             np.mean,
                             np.mean,
                             embeddings,
                             reciprocal,
                             10,
                             k=1,
                             use_focus=False)

    # Baseline space with 10000 words.
    txt = list(chain.from_iterable(txt))
    embeddings = baseline(txt, 10000)

    baseline = experiment(parsed_train,
                          gold_chunks_train,
                          parsed_test,
                          gold_chunks_test,
                          np.mean,
                          np.mean,
                          embeddings,
                          reciprocal,
                          0,
                          k=1,
                          use_focus=True)

    scores_knn = {'focus': knn_focus,
                  'full': knn_full,
                  'context': knn_context,
                  'baseline': baseline}

    scores = {}

    for k, v in scores_knn.items():

        t, p = zip(*v)
        scores[k] = evaluate_k(t, p, None)

    json.dump(scores, open("scores_transfer.json", 'w'))
    json.dump(scores_knn, open("knn_transfer.json", 'w'))
