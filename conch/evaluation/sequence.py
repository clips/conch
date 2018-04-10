"""Simple evaluation script."""
from .utils import bio_to_index
from collections import Counter, defaultdict
from itertools import chain


def precision_recall_dict(tp, fp, fn, average=None):
    """Calculate precision and recall for dictionaries."""
    if average is None or average == "macro":

        p_denominator = Counter(tp) + Counter(fp)
        r_denominator = Counter(tp) + Counter(fn)

        precision = {k: v / p_denominator[k] for k, v in tp.items()}
        recall = {k: v / r_denominator[k] for k, v in tp.items()}
        f1 = {k: 2 * ((precision[k] * recall[k]) / (precision[k] + recall[k]))
              for k in precision.keys()}

        if average == "macro":

            precision = sum(precision.values()) / len(precision)
            recall = sum(recall.values()) / len(recall)
            f1 = sum(f1.values()) / len(f1)

    elif average == "micro":

        p_denominator = sum(tp.values()) + sum(fp.values())
        r_denominator = sum(tp.values()) + sum(fn.values())

        precision = sum(tp.values()) / p_denominator
        recall = sum(tp.values()) / r_denominator

        f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1


def eval_sequence(gold_bio, pred_bio, exact=True):
    """Evaluate sequences on the basis of BIO strings."""
    assert len(gold_bio) == len(pred_bio)

    if not isinstance(gold_bio[0], list):
        gold_bio = [gold_bio]
        pred_bio = [pred_bio]

    for x, y in zip(gold_bio, pred_bio):
        assert len(x) == len(y)

    gold_index = [set(x) for x in bio_to_index(gold_bio)]
    pred_index = [set(x) for x in bio_to_index(pred_bio)]

    tp, fp, fn = Counter(), Counter(), Counter()

    if exact:

        for g, p in zip(gold_index, pred_index):

            tp_, fp_, fn_ = eval_triples_exact(g, p)
            tp.update(tp_)
            fp.update(fp_)
            fn.update(fn_)

    else:

        for g, p, gb, pb in zip(gold_index, pred_index, gold_bio, pred_bio):

            tp_, fp_, fn_ = eval_inexact(g, p, gb, pb)
            tp.update(tp_)
            fp.update(fp_)
            fn.update(fn_)

    # Sanity checks.
    num_pred = len(list(chain.from_iterable(pred_index)))
    num_gold = len(list(chain.from_iterable(gold_index)))
    assert sum(tp.values()) + sum(fp.values()) == num_pred
    assert sum(tp.values()) + sum(fn.values()) == num_gold

    return tp, fp, fn


def eval_triples_exact(gold_index, pred_index):
    """Exact evaluation on triples."""
    tp = Counter([x[-1] for x in gold_index.intersection(pred_index)])
    fp = Counter([x[-1] for x in pred_index - gold_index])
    fn = Counter([x[-1] for x in gold_index - pred_index])

    return tp, fp, fn


def eval_inexact(gold_index, pred_index, gold_bio, pred_bio):
    """Perform inexact evaluation."""
    tp_p = defaultdict(int)
    tp_g = defaultdict(int)

    for begin, end, label in pred_index:
        labels = set([x.split("-")[-1] for x in gold_bio[begin:end]
                      if x != "O"])
        if label in labels:
            tp_p[label] += 1

    for begin, end, label in gold_index:
        labels = set([x.split("-")[-1] for x in pred_bio[begin:end]
                      if x != "O"])
        if label in labels:
            tp_g[label] += 1

    tp = {k: min([tp_p[k], v]) for k, v in tp_g.items()}
    fn = {k: v - tp[k] if k in tp else v for k, v in
          Counter([x[-1] for x in gold_index]).items()}
    fp = {k: v - tp[k] if k in tp else v for k, v in
          Counter([x[-1] for x in pred_index]).items()}

    return tp, fp, fn


def sigf_evaluation_output(gold_bio, pred_bio, out_path, exact=True):
    """
    Produce an output file with the output format for sigf.

    https://nlpado.de/~sebastian/software/sigf.shtml
    Sigf takes a triple of three numbers for each document and a given
    system: (TP, # predicted, # gold)

    Parameters
    ==========
    gold_bio : list of list
        A list of list of BIO tags
    pred_bio : list of list
        A list of list of BIO tags
    out_path : str
        The path to write the sigf output to.

    """
    if not isinstance(gold_bio[0], list):
        gold_bio = [gold_bio]
    if not isinstance(pred_bio[0], list):
        pred_bio = [pred_bio]

    gold_index = [set(x) for x in bio_to_index(gold_bio)]
    pred_index = [set(x) for x in bio_to_index(pred_bio)]

    f = open(out_path, 'w')

    if exact:

        for g, p in zip(gold_index, pred_index):

            tp, fp, fn = eval_triples_exact(g, p)
            tp = sum(tp.values())
            fp = sum(fp.values())
            fn = sum(fn.values())
            f.write("{} {} {}\n".format(tp, fp + tp, fn + tp))

    else:

        for g, p, gb, pb in zip(gold_index, pred_index, gold_bio, pred_bio):

            tp, fp, fn = eval_inexact(g, p, gb, pb)
            tp = sum(tp.values())
            fp = sum(fp.values())
            fn = sum(fn.values())
            f.write("{} {} {}\n".format(tp, fp + tp, fn + tp))
