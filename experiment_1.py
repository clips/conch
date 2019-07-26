"""
Run intrinsic evaluation experiments from the paper.

Running this code will replicate experiment 1 from the paper.
Because of the low run-time for these experiments, we run both the Perfect
and parsed experiments at the same time.
"""
import json

from conch.evaluation.intrinsic import evaluate_intrinsic
from conch.evaluation.utils import evaluate_k
from reach import Reach
from conch.conch import compose, reciprocal
from conch.preprocessing.baseline import baseline
from itertools import chain


def experiment(parsed,
               gold_chunks,
               embeddings,
               context_function,
               window,
               k,
               use_focus=True):
    """Run an experiment with intrinsic evaluation."""
    _, np_chunks = zip(*parsed)

    phrase_embeddings = compose(parsed,
                                window=window,
                                embeddings=embeddings,
                                context_function=context_function,
                                use_focus=use_focus,
                                norm=True)

    result = evaluate_intrinsic(gold_chunks,
                                np_chunks,
                                phrase_embeddings,
                                k=k)

    return result


if __name__ == "__main__":

    scores = {}

    gold = json.load(open("data/beth_gold.json"))
    gold = list(zip(*sorted(gold.items())))[1]

    txt, gold_chunks = zip(*gold)

    data = json.load(open("data/beth_uima.json"))
    data = list(zip(*sorted(data.items())))[1]

    # Sanity check
    for a, b in zip(data, gold):
        assert len(a[0]) == len(b[0])

    embeddings = Reach.load("", unk_word="UNK")

    scores = {}

    focus = experiment(data,
                       gold_chunks,
                       embeddings,
                       reciprocal,
                       0,
                       k=100,
                       use_focus=True)

    full = experiment(data,
                      gold_chunks,
                      embeddings,
                      reciprocal,
                      10,
                      k=100,
                      use_focus=True)

    context = experiment(data,
                         gold_chunks,
                         embeddings,
                         reciprocal,
                         10,
                         k=100,
                         use_focus=False)

    full_perfect = experiment(gold,
                              gold_chunks,
                              embeddings,
                              reciprocal,
                              10,
                              k=100,
                              use_focus=True)

    focus_perfect = experiment(gold,
                               gold_chunks,
                               embeddings,
                               reciprocal,
                               0,
                               k=100,
                               use_focus=True)

    context_perfect = experiment(gold,
                                 gold_chunks,
                                 embeddings,
                                 reciprocal,
                                 10,
                                 k=100,
                                 use_focus=False)

    # Baseline space with 10000 words.
    txt = list(chain.from_iterable(txt))
    embeddings = baseline(txt, 10000)

    for a, b in zip(data, gold):
        assert len(a[0]) == len(b[0])

    baseline = experiment(data,
                          gold_chunks,
                          embeddings,
                          reciprocal,
                          0,
                          k=100,
                          use_focus=True)

    baseline_perfect = experiment(gold,
                                  gold_chunks,
                                  embeddings,
                                  reciprocal,
                                  0,
                                  k=100,
                                  use_focus=True)

    scores_knn = {'focus': focus,
                  'full': full,
                  'context': context,
                  'focus_perfect': focus_perfect,
                  'full_perfect': full_perfect,
                  'context_perfect': context_perfect,
                  'baseline': baseline,
                  'baseline_perfect': baseline_perfect}

    scores = {}

    for k, v in scores_knn.items():

        t, p = zip(*v)
        scores[k] = evaluate_k(t, p, None)

    json.dump(scores, open("scores_intrinsic_norm.json", 'w'))
    json.dump(scores_knn, open("knn_intrinsic_norm.json", 'w'))
