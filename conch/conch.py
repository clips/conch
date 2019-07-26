"""conch composes word vectors."""
from __future__ import division

import numpy as np
import regex as re

from reach import Reach

removal = re.compile(r"[\d]+\.\s", re.UNICODE)


def identity(x, **kwargs):
    """Identity function."""
    return x


def reciprocal(x):
    """
    Weigh matrices by a reciprocal from some word.

    Reciprocal function: takes a matrix, and returns
    the matrix divided by the reciprocal of its index + factor.
    Increasing the factor diminishes the influence of the whole
    matrix on some other set of vectors, but has a negligible
    effect on the matrix itself.

    ex. in one dimension:

        input:      [4,    4,    4,    4,    4  ]
        index:      [0,    1,    2,    3 ,   4  ]
        reciprocal: [1,    0.5,  0.33, 0.25, 0.2]
        result:     [4,    2,    1.32, 1,    0.8]

    This function is used for weighting contexts.

    Parameters
    ==========
    x : np array or list
        The input data which is weighted.
    factor : float
        A constant which is added to the reciprocal before it is divided.
        Increasing the factor will cause the effect of reciprocal weighting to
        be lessened.

    Returns
    =======
    weighted : np.array
        A weighted version of the input array.

    """
    # Create the reciprocal
    z = np.reciprocal(np.arange(1, len(x)+1, dtype=np.float32))

    if type(x) == list:
        x = np.array(x)

    # Weigh the original matrix by the reciprocal.
    return x * z[:, None]


def compose(documents,
            embeddings,
            window,
            context_function,
            use_focus=True,
            norm=False):
    """
    Map phrases from sentences to vectors.

    Parameters
    ==========
    documents : list of lists
        A list of lists, where each sublist contains 2 lists of the same
        length, where the first list contains the tokens of a text, and
        the second list contains the BIO of the NP chunks for said text.
    embeddings : Reach
        A reach instance which contains the embeddings you want to use to
        vectorize.
    window : int
        The window size to use.
    context_function : function
        The function which is used to weigh the contexts. Must take a 2D
        matrix and return a 2D matrix of the same shape.
    use_focus : bool, optional, default True
        Whether to vectorize the focus word.
    norm : bool, optional, default False
        Whether to use the unit vectors to compose.

    Returns
    =======
    phrases : Reach
        A reach instance containing the phrases and their vectors.

    """
    bio_regex = re.compile(r"BI*")

    phrases, vectors = [], []

    for idx, (txt, bio) in enumerate(documents):

        txt = " ".join(txt).lower().split()
        bio = "".join([x.split("-")[0] for x in bio])
        for t in bio_regex.finditer(bio):
            b, e = t.span()
            phrase_string, vector = create_phrase_vector(txt,
                                                         b,
                                                         e,
                                                         window,
                                                         embeddings,
                                                         np.mean,
                                                         np.mean,
                                                         context_function,
                                                         use_focus,
                                                         norm)

            # Phrase string needs to be augmented with index to make
            # the dictionary mapping not overwrite itself.
            phrase_string = "{}-{}".format(phrase_string, len(phrases))
            phrases.append(phrase_string)
            vectors.append(vector)

    return Reach(vectors, phrases)


def create_phrase_vector(doc,
                         begin,
                         end,
                         window,
                         embeddings,
                         f1,
                         f2,
                         context_function,
                         use_focus,
                         norm):
    """Create a phrase vector by vectorizing the left and right contexts."""
    if use_focus:
        phrase = doc[begin:end]
    else:
        phrase = []
    # Create windows.
    if window > 0:
        right_window = doc[end:end+window]
        left_window = doc[begin-window:begin]
    else:
        right_window, left_window = [], []

    # Vectorize the context.
    vector = _vectorize_context(phrase,
                                left_window,
                                right_window,
                                embeddings,
                                f1,
                                f2,
                                context_function,
                                norm)

    return ("{}-{}-{}".format(left_window[::-1], phrase, right_window),
            vector)


def _vectorize_context(phrase,
                       left_window,
                       right_window,
                       embeddings,
                       f1,
                       f2,
                       context_function,
                       norm):
    """Vectorize the context based on two functions."""
    if phrase:
        phrase_vec = embeddings.vectorize(phrase,
                                          remove_oov=False,
                                          norm=norm)
        phrase_vec = f1(phrase_vec, axis=0)
    else:
        phrase_vec = np.zeros(embeddings.size)
    if left_window:
        left_vec = embeddings.vectorize(left_window,
                                        remove_oov=False,
                                        norm=norm)
        left_vec = f1(context_function(left_vec), axis=0)
    else:
        left_vec = np.zeros(embeddings.size)
    if right_window:
        right_vec = embeddings.vectorize(right_window,
                                         remove_oov=False,
                                         norm=norm)
        right_vec = f1(context_function(right_vec), axis=0)
    else:
        right_vec = np.zeros(embeddings.size)

    vector = f2([left_vec, phrase_vec, right_vec], axis=0)

    return vector
