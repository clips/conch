"""Intrinsic evaluation."""
import numpy as np

from collections import Counter
from reach import Reach
from tqdm import tqdm
from .utils import bio_to_index


def evaluate_transfer(gold_bio,
                      phrase_bio,
                      gold_bio_test,
                      phrase_bio_test,
                      train_embeddings,
                      test_embeddings,
                      k=10,
                      batch_size=250):
    """
    Do a transfer experiment between corpora.

    Equivalent to doing KNN with a test set.

    Parameters
    ==========
    gold_bio : list of lists of strings
        The BIO strings of the gold standard data. We use one string per
        document. This is used to align the extracted chunks to the gold
        standard, and determine which chunks are illegal.
    phrase_bio : list of lists of strings
        The BIO strings of the chunked data. NOTE: These BIO strings are
        produced by an NP chunker, and are used together with the gold BIO
        to determine which chunks get assigned which label.
    gold_bio_test : list of lists of strings
        Same as gold_bio, but for the test set.
    phrase_bio_test : list of lists of strings
        Same as phrase_bio, but for the test set.
    train_embeddings : Reach
        The phrase embeddings on the train set. Must be aligned with the
        phrase_bio, which means that there must be as many concepts in the BIO
        string as in the training set.
    test_embeddings : Reach
        The phrase embeddings on the test set. Must be aligned with the
        phrase_bio, which means that there must be as many concepts in the BIO
        string as in the test set.
    k : int, optional, default 10
        The k nearest neighbors to consider in the knn experiment.
    batch_size : int, optional, default 250
        The batch size to use.

    Returns
    =======
    neighbors : list of lists
        The labels of the k nearest neighbors, which can then be used in
        subsequent scoring functions.

    """
    train_embeddings, words2label, _, _ = label_chunks(gold_bio,
                                                       phrase_bio,
                                                       train_embeddings)

    test_embeddings, _, phrase_labels, results = label_chunks(gold_bio_test,
                                                              phrase_bio_test,
                                                              test_embeddings)

    return produce_eval(phrase_labels,
                        test_embeddings,
                        train_embeddings,
                        words2label,
                        k,
                        results,
                        batch_size,
                        0)


def evaluate_intrinsic(gold_bio,
                       phrase_bio,
                       embeddings,
                       k=10,
                       batch_size=250):
    """
    Do a transfer experiment between corpora.

    Equivalent to doing KNN without a test set.

    Parameters
    ==========
    gold_bio : list of lists of strings
        The BIO strings of the gold standard data. We use one string per
        document. This is used to align the extracted chunks to the gold
        standard, and determine which chunks are illegal.
    phrase_bio : list of lists of strings
        The BIO strings of the chunked data. NOTE: These BIO strings are
        produced by an NP chunker, and are used together with the gold BIO
        to determine which chunks get assigned which label.
    embeddings : Reach
        The phrase embeddings on the test set. Must be aligned with the
        phrase_bio, which means that there must be as many concepts in the BIO
        string as in the test set.
    k : int, optional, default 10
        The k nearest neighbors to consider in the knn experiment.
    batch_size : int, optional, default 250
        The batch size to use.

    Returns
    =======
    neighbors : list of lists
        The labels of the k nearest neighbors, which can then be used in
        subsequent scoring functions.

    """
    embeddings, words2label, phrase_labels, results = label_chunks(gold_bio,
                                                                   phrase_bio,
                                                                   embeddings)

    results = [(x, [y] * k) for x, y in results]

    return produce_eval(phrase_labels,
                        embeddings,
                        embeddings,
                        words2label,
                        k,
                        results,
                        batch_size,
                        1)


def label_chunks(gold_bio,
                 phrase_bio,
                 embeddings):
    """
    Find a label for each phrase chunk based on the gold chunks.

    Each phrase chunk which does not correctly overlap with a gold chunk
    is pruned from the embedding space and added as a false positive.

    Parameters
    ==========
    gold_bio : list of string
        The token-level BIO string for the gold standard data. Must include
        classes on the B and I labels (e.g. B-Test, I-test).
    phrase_bio : list of string
        The token-level BIO string for the phrase data. Does not include
        any classes on the B and I labels. (e.g. B and I instead of B-test).
    embeddings : Reach
        The embedding space for the phrases.

    Returns
    =======
    pruned_embeddings : Reach
        The embedding space with any false positive phrases removed.
    word2label : dict
        Dictionary mapping from the name of each phrase to a label.
    chunk_labels : np.array
        An aligned list from phrases to labels.
    results : list of tuples
        An intermediate list of false positives and false negatives constructed
        during matching the gold and phrase chunks.

    """
    # Create a list of (start, end, label) tuples from BIO.
    phrase_chunks = bio_to_index(phrase_bio)
    gold_chunks = bio_to_index(gold_bio)
    phrase_labels, results = link_chunks_to_gold(phrase_chunks, gold_chunks)

    # False positives get assigned the label "o", so they need to be removed.
    allowed = [i for i, v in enumerate(phrase_labels) if v != "o"]

    if results:
        t, _ = zip(*results)
        print("Num false neg: {0}".format(Counter(t)))

    # We assume alignment between chunks and words.
    vectors = embeddings.norm_vectors[allowed]
    chunk_labels = np.array(phrase_labels)[allowed]
    words = [embeddings.indices[x] for x in allowed]
    words2label = {embeddings.indices[x]: chunk_labels[idx]
                   for idx, x in enumerate(allowed)}

    pruned_embeddings = Reach(vectors, words)

    return pruned_embeddings, words2label, chunk_labels, results


def produce_eval(phrase_labels,
                 embeddings,
                 reference_embeddings,
                 words2label,
                 k=1,
                 results=((), ()),
                 batch_size=250,
                 add=1):
    """Produce the actual evaluation."""
    for batch in tqdm(range(0, len(embeddings.norm_vectors), batch_size)):

        labels = phrase_labels[batch:batch+batch_size]

        batch = embeddings.norm_vectors[batch:batch+batch_size]

        # Compute the distances from the current batch to all other vectors.
        r = reference_embeddings.nearest_neighbor(batch, num=k+add)
        for result, label, vec in zip(r, labels, batch):
            if not vec.any():
                results.append((label, ["o"] * k))
                continue

            closest = [words2label[x[0]] for x in result[add:]]
            results.append((label, closest))

    return results


def calculate_shared(a, b):
    """
    Calculate the overlap between a chunk and a list of chunks.

    The overlap is only calculated from a to b.

    Parameters
    ==========
    a : tuple (begin, end)
        Represents the begin and end coordinates of a chunk.
    b : list of tuples
        A list of tuples containing the begin and end coordinates of chunks.

    Returns
    =======
    shr : list
        The chunks from b which overlap with a.

    """
    start_1, end_1 = a

    shr = []

    for idx, (start, end) in enumerate(b):

        # If the start of the a chunk is larger than the end of the b chunk,
        # quit. If the start of the b chunk is larger than the end of the a
        # chunk, also quit.
        if end < start_1 or end_1 < start:
            continue

        # Complete overlap
        if start_1 == start and end_1 == end:
            shr.append(idx)
            continue

        # At least partial overlap: start of a is within bounds of bs
        if start <= start_1 < end:
            shr.append(idx)
            continue

        # At least partial overlap: end of a is within bounds of b
        if start < end_1 <= end:
            shr.append(idx)
            continue

        # b is contained within a
        if start_1 <= start and end_1 >= end:
            shr.append(idx)
            continue

        # a is contained within b
        if start <= start_1 and end >= end_1:
            shr.append(idx)
            continue

    return shr


def overlap(gold_chunks, phrase_chunks):
    """
    Calculate the overlap between a list of phrase and gold chunks.

    Parameters
    ==========
    gold_chunks : list of tuple
        A list of (begin, end, label) tuples.
    phrase_chunks : list of tuple
        A list of (begin, end, label) tuples.

    Returns
    =======
    results : tuple
        A tuple, the first item of which is the intermediate results, which
        contains false negatives and false positives because of multiple
        overlap. The second item is a list of labels of the phrase chunks.

    """
    results = []

    # Track all a chunks
    # A chunks start being untouched
    touched = [False] * len(gold_chunks)
    # Track all b chunks
    # B chunks start as np labels
    labels = ["np"] * len(phrase_chunks)

    p_without_label = [(x, y) for x, y, z in phrase_chunks]

    for idx, (start, end, label) in enumerate(gold_chunks):

        shr = calculate_shared((start, end), p_without_label)

        if len(shr) == 1:
            for x in shr:
                labels[x] = label

        elif len(shr) > 1:
            for x in shr:
                labels[x] = 'o'

        touched[idx] = len(shr) == 1

    g_without_label = [(x, y) for x, y, z in gold_chunks]

    for idx, (start, end, label) in enumerate(phrase_chunks):

        # Calculate the overlap between a chunk and the gold labels
        shr = calculate_shared((start, end), g_without_label)

        # If the overlap is 1 or 0, the label is maintained, else it is
        # set to "o", and counted as a false positive.
        labels[idx] = labels[idx] if len(shr) <= 1 else "o"

        # Because the gold labels no longer occur with a chunk, they need
        # to be set to False.
        if len(shr) > 1:

            for x in shr:
                touched[x] = False

    # All untouched gold chunks are false negatives.
    for x in [x for x, y in zip(gold_chunks, touched) if not y]:
        res = (x[2], "np")
        results.append(res)

    # We need a label for each predicted chunk.
    assert(len(labels) == len(phrase_chunks))
    # We need the number of illegal and legal pred chunks to be equal to the
    # number of gold chunks.
    total_len = len([x for x in labels if x not in ["o", "np"]]) + len(results)
    assert(total_len == len(gold_chunks))

    return results, labels


def link_chunks_to_gold(phrase_chunks, gold_chunks):
    """
    Label a list of chunks by comparing them to gold standard coordinates.

    This evaluation is inexact: it will assign a label if there is a overlap
    between a chunk and a gold label. In terms of other constraints, the
    evaluation is strict:

        - If a chunk overlaps with multiple golds, it is counted
          as being incorrect (we do not assign a label).

        - If a gold standard BIO chunk overlaps with
          multiple chunks, it is counted as a false negative.

    This is not done for mere convenience: it is the only
    way our system is correct: other ways of assigning labels always
    result in there being a variable number of gold standard chunks
    in the end, which is impossible.

    Parameters
    ==========
    phrase_chunks : dict of list of tuples (begin, end, label)
        A dictionary for each document. Each document is represented by a
        list of tuples representing the phrase boundaries for the phrases in
        that document.
    gold_chunks : dict of list of tuples (begin, end, label)
        Same as phrase_chunks, but based on gold standard data.

    Returns
    =======
    chunk_labels : list of str
        A list containing the label for each phrase chunk.
    results : list of tuple
        A list containing the intermediate true positives and false negatives.

    """
    results, chunk_labels = [], []

    for gold, phrase in zip(gold_chunks, phrase_chunks):

        res, labels = overlap(gold, phrase)

        # true and pred are labels of chunks for
        # which we could not obtain a guess
        # because of overlap with multiple chunks
        # or vice versa.
        results.extend(res)
        chunk_labels.extend(labels)

    return chunk_labels, results
