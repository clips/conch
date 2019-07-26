"""
Extrinsic evaluation on the i2b2 data.

Experiment 3 in the paper.
Set the boolean flag Perfect to True if you want to try perfect chunking.
"""
import json

from itertools import chain
from conch.evaluation.extrinsic import eval_extrinsic
from conch.preprocessing.baseline import baseline
from conch.preprocessing.concept_vectors import create_concepts
from reach import Reach
from conch.conch import compose, reciprocal
from conch.evaluation.utils import to_conll


if __name__ == "__main__":

    # Set this flag to true to replicate the perfect chunking setting
    # in experiment 3.
    perfect = False

    gold = json.load(open("data/test_gold.json"))
    gold = list(zip(*sorted(gold.items())))[1]

    if perfect:
        data = json.load(open("data/test_gold.json"))
    else:
        data = json.load(open("data/test_uima.json"))
    data = list(zip(*sorted(data.items())))[1]

    txt, gold_bio = zip(*gold)
    _, data_bio = zip(*data)

    embeddings = Reach.load("", unk_word="UNK")
    concept_reach = Reach.load_fast_format("data/concept_vectors")
    concept_labels = json.load(open("data/concept_names2label.json"))

    gold_bio = list(chain.from_iterable(gold_bio))

    results_bio = {}

    r_phrases = compose(data,
                        window=0,
                        embeddings=embeddings,
                        context_function=reciprocal)

    pred_bio_focus = eval_extrinsic(list(chain.from_iterable(data_bio)),
                                    r_phrases,
                                    concept_reach,
                                    concept_labels,
                                    250)

    r_phrases = compose(data,
                        window=10,
                        embeddings=embeddings,
                        context_function=reciprocal)

    pred_bio_full = eval_extrinsic(list(chain.from_iterable(data_bio)),
                                   r_phrases,
                                   concept_reach,
                                   concept_labels,
                                   250)

    txt = list(chain.from_iterable(txt))
    baseline_embeddings = baseline(txt, 10000)
    concept_baseline, concept_labels = create_concepts(baseline_embeddings,
                                                       include_np=True)

    r_phrases = compose(data,
                        window=0,
                        embeddings=baseline_embeddings,
                        context_function=reciprocal)

    pred_bio_baseline = eval_extrinsic(list(chain.from_iterable(data_bio)),
                                       r_phrases,
                                       concept_baseline,
                                       concept_labels,
                                       250)

    json.dump(results_bio, open("results/knn_test_extrinsic.json", 'w'))

    to_conll(pred_bio_focus, gold_bio, "results/test_focus_model.conll")
    to_conll(pred_bio_full, gold_bio, "results/test_full_model.conll")
    to_conll(pred_bio_baseline, gold_bio, "results/test_baseline.conll")
