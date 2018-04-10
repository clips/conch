"""Analysis of the quality of the chunker with regard to the gold standard."""
import json
from conch.evaluation.sequence import eval_sequence, precision_recall_dict
from itertools import chain
from conch.evaluation.utils import bio_to_index
from collections import Counter


if __name__ == "__main__":

    gold = json.load(open("data/beth_gold.json"))
    gold = list(zip(*sorted(gold.items())))[1]

    data = json.load(open("data/beth_uima.json"))
    data = list(zip(*sorted(data.items())))[1]

    txt, gold_bio = zip(*gold)
    _, data_bio = zip(*data)

    gold_bio = list(chain.from_iterable(gold_bio))
    data_bio = list(chain.from_iterable(data_bio))

    gold_bio = ["{}-NP".format(x[0]) if x != "O" else x for x in gold_bio]

    gold_indices = bio_to_index([gold_bio])[0]
    data_indices = bio_to_index([data_bio])[0]

    lens = Counter()
    words = Counter()

    keep = []

    tp, fp, fn = eval_sequence(gold_bio, data_bio)
    exact = precision_recall_dict(tp, fp, fn)
    tp, fp, fn = eval_sequence(gold_bio, data_bio, exact=False)
    inexact = precision_recall_dict(tp, fp, fn)

    for (x, y, _) in data_indices:
        b = gold_bio[x:y]
        s = set([x.split("-")[-1] for x in gold_bio[x:y]])
        if "NP" in s and "O" in s:
            if b[0].startswith("I"):
                keep.append(((x, y), (x, x+b.index("O"))))
            else:
                for idx, s in enumerate(b):
                    if s.startswith("B"):
                        s = x + idx
                        break
                    try:
                        keep.append(((x, y), (s, x + b[idx:].index("O"))))
                    except ValueError:
                        keep.append(((x, y), (s, y)))

    for (a, b), (c, d) in keep:
        lens.update([len(txt[a:b]) - len(txt[c:d])])
        words.update(set(txt[a:b]) - set(txt[c:d]))
