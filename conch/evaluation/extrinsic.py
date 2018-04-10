"""Evaluation against a set of concept labels."""
import numpy as np

from tqdm import tqdm
from .utils import bio_to_index


def eval_extrinsic_label(vectors, concepts, labels, batch_size):
    """
    Evaluate the set of composed vectors against a set of concept vectors.

    Parameters
    ==========
    vectors : Reach
        A reach instance which contains the composed vectors.
    concepts : Reach
        A reach instance which contains the composed concept vectors.
    labels : list of string
        A list of labels for each concept, which are used to assign labels.
        Must be equal to the number of concepts.
    batch_size : int
        The batch size to use during processing.

    Returns
    =======
    results : list
        A label for each chunk.

    """
    results = []
    nones = 0

    for batch in tqdm(range(0, len(vectors.norm_vectors), batch_size)):

        batch = vectors.norm_vectors[batch:batch+batch_size]

        # Compute the distances from the current batch to all other vectors.
        res = concepts.nearest_neighbor(batch, num=1)
        for result, vec in zip(res, batch):
            if not np.any(vec):
                results.append("np")
                nones += 1
                continue

            results.append(labels[result[0][0]])

    assert(len(results) == len(vectors.norm_vectors))
    return results


def eval_extrinsic(chunk_bio,
                   vectors,
                   concepts,
                   concept_labels,
                   batch_size):
    """
    Produce a BIO sequence of labels given a BIO sequence of Phrase chunks.

    A class is produced for each chunk, which is then inserted in the BIO
    sequence at the appropriate location. For chunks which have the label
    "np", no chunk is inserted in the BIO sequence.

    Parameters
    ==========
    chunk_bio : list of strings
        A list of BIO tags.
    vectors : Reach
        A reach instance which contains the composed chunk vectors.
    concepts : Reach
        A reach instance which contains the composed concept vectors.
    labels : list of string
        A list of labels for each concept, which are used to assign labels.
        Must be equal to the number of concepts.
    batch_size : int
        The batch size to use during processing.

    Returns
    =======
    new_bio : list of string
        A list of BIO tags, with the length of the original BIO sequence.

    """
    new_bio = ["O"] * len(chunk_bio)
    results = eval_extrinsic_label(vectors,
                                   concepts,
                                   concept_labels,
                                   batch_size)

    # bio_to_index produces a dict, and expects multiple sequences
    # so we pass a list, and take the first element of the dict.
    chunk_indices = bio_to_index([chunk_bio])[0]

    assert len(results) == len(chunk_indices)

    for (begin, end, _), label in zip(chunk_indices, results):
        if label == "np":
            continue
        for idx in range(0, end-begin):
            if idx == 0:
                new_bio[begin] = "B-{}".format(label)
            else:
                new_bio[begin + idx] = "I-{}".format(label)

    return new_bio
