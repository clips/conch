"""Extract gold chunks from the i2b2 dataset."""
import os
import json
from itertools import chain, combinations
from collections import defaultdict

from glob import iglob


def _single_overlap(a, b):
    """Check overlap between two chunks."""
    b_1, e_1 = a
    b_2, e_2 = b
    if a == b:
        return True
    if b_1 <= b_2 <= e_1:
        return True
    if b_2 <= b_1 <= e_2:
        return True
    if b_1 <= e_2 <= e_1:
        return True
    if b_2 <= e_1 <= e_2:
        return True
    return False


def _check_overlap(chunks):
    """Check for overlap between chunks."""
    index = list(range(len(chunks)))
    overlaps = []
    for x, y in combinations(index, 2):
        if _single_overlap(chunks[x], chunks[y]):
            overlaps.append({x, y})

    return overlaps


def extract_chunks(text_path, con_path, remove_overlap=False):
    """Extract chunks from a matching set of .txt and .con files."""
    con = open(con_path).readlines()
    text = [x.split() for x in open(text_path).readlines()]
    bio = [["O"] * len(x) for x in text]

    bio_dict = defaultdict(list)

    for x in con:
        rest, tag = x.split("||")
        tag = tag.split("=")[-1][1:-2]
        begin, end = rest.split()[-2:]
        line_no, begin = begin.split(":")
        end = end.split(":")[1]
        line_no, begin, end = int(line_no)-1, int(begin), int(end)
        bio_dict[line_no].append((begin, end, tag))

    for k, v in bio_dict.items():

        overlaps = _check_overlap([(b, e) for b, e, _ in v])
        to_remove = []

        for a, b in overlaps:
            if v[a] == v[b]:
                continue
            b_1, e_1, _ = v[a]
            b_2, e_2, _ = v[b]

            id_1 = "span: {}".format(v[a])
            id_2 = "span: {}".format(v[b])
            span_txt_1 = " ".join(text[k][b_1: e_1+1])
            span_txt_2 = " ".join(text[k][b_2: e_2+1])
            print("Warning: the following spans are non-identical, "
                  "but have some overlap: "
                  "{}, {}, {}, {}".format(filename, k, id_1, id_2))
            print("A: {}".format(span_txt_1))
            print("B: {}".format(span_txt_2))
            if remove_overlap:
                if e_1 - b_1 > e_2 - b_2:
                    to_remove.append(b)
                    print("Removed {}".format(id_2, span_txt_2))
                else:
                    to_remove.append(a)
                    print("Removed {}, {}".format(id_1, span_txt_1))
        to_remove = set(to_remove)
        v = [x for idx, x in enumerate(v) if idx not in to_remove]
        line_no = k
        for begin, end, tag in v:
            for x in range((end-begin)+1):
                if x == 0:
                    t = "B"
                else:
                    t = "I"
                bio[line_no][begin+x] = "{}-{}".format(t, tag)

    return (" ".join([" ".join(x) for x in text]).split(),
            list(chain.from_iterable(bio)))


if __name__ == "__main__":

    # TODO: these paths are for convenience,
    # will be replaced by argparsed paths.
    base_path = ""
    beth_path = os.path.join(base_path, "beth/*.txt")
    partners_path = os.path.join(base_path, "partners/*.txt")
    test_path = os.path.join(base_path, "test/*.txt")

    train = {}
    test = {}
    beth = {}
    partners = {}

    for filename in iglob(beth_path):

        without_ext = os.path.splitext(filename)[0]
        key = os.path.splitext(os.path.split(filename)[-1])[0]
        c = extract_chunks(filename, without_ext + ".con")
        train[key] = c
        beth[key] = c

    for filename in iglob(partners_path):

        without_ext = os.path.splitext(filename)[0]
        key = os.path.splitext(os.path.split(filename)[-1])[0]
        c = extract_chunks(filename, without_ext + ".con")
        train[key] = c
        partners[key] = c

    for filename in iglob(test_path):

        without_ext = os.path.splitext(filename)[0]
        key = os.path.splitext(os.path.split(filename)[-1])[0]
        test[key] = extract_chunks(filename, without_ext + ".con")

    json.dump(train, open("data/train_gold.json", 'w'))
    json.dump(test, open("data/test_gold.json", 'w'))
    json.dump(beth, open("data/beth_gold.json", 'w'))
    json.dump(partners, open("data/partners_gold.json", 'w'))
