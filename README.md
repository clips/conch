# ConCH

Unsupervised concept extraction from clinical text.

This is the code associated with the publication: `Unsupervised concept extraction from clinical text through semantic composition`, published in the [Journal of Biomedical Informatics](https://www.sciencedirect.com/science/article/pii/S1532046419300383).

If you use this code, please cite:

```
@article{tulkens2019unsupervised,
  title={Unsupervised Concept Extraction from Clinical Text through Semantic Composition},
  author={Tulkens, St{\'e}phan and {\v{S}}uster, Simon and Daelemans, Walter},
  journal={Journal of Biomedical Informatics},
  pages={103120},
  year={2019},
  publisher={Elsevier}
}
```

`ConCH` (Concept CHecker) extracts concepts by first extracting noun phrases from a corpus using a chunker or a parser. These noun phrases are then turned into vector representations through composition of their constituent word vectors. In the paper, we use the `mean` function as a composition function, but other functions can also be used.

These phrase representations are then compared to similar representations of concepts, which are composed from the textual descriptions and names given to concepts in an ontology.

The main take-aways from the paper are that the addition of context, in the form of windows, in these representations helps very little, while the phrase representations themselves are good enough to extract a variety of concepts.

## License

GPL-3.0

## requirements

* numpy
* sklearn
* reach
* tqdm
* lxml
* regex

## Usage

Conch requires the following to work.

* A set of word vectors
* A mapping from concepts to their descriptions or strings
* A parser

If you want to replicate the experiments in the paper, you will need access to the [i2b2-2010](https://www.i2b2.org/NLP/DataSets/Main.php) challenge corpus.

If you have access to the i2b2-2010 challenge corpus, please run all the preprocessing scripts in `conch.preprocessing` to extract noun phrases, and convert the gold standard data to `IOB` format. We currently offer a conversion script from `UIMA` `XML` format to `IOB` format. If you use another parser or chunker, you will have to write your own converter.

Concept representations are also created using a preprocessing script. The input to this script is a dictionary (we use a JSON file), with the UMLS CUIs as keys, and the descriptions as lists of strings.

## Example

Conch expects a list of tuples of lists as input.
The first list is the tokenized text of your input data, while the second list is a set of IOB tags.
Here's an example:

```python
data = [(["the", "cat", "walked", "home"],
         ["B-NP", "I-NP", "O", "B-NP"]),
        (["she", "had", "some", "milk"],
         ["B-NP", "O", "B-NP", "I-NP"])],
         ...]
```


In all our experiments we used whole documents instead of sentences, but you can pick whichever you want.
Note that it is important that there are exactly as many IOB tags as (tokenized) words in every sentence.

Using word embeddings, these sentences can be composed into phrases.

```python
from reach import Reach
from conch import compose

r = Reach.load("my_embeddings.vec")
phrases = compose(data,
                  f1=np.mean,
                  f2=np.mean,
                  embeddings=r,
                  window=5,
                  context_function=lambda x: x)

# Phrases is an embedding space containing phrases that can be saved.
print(phrases.items)
>>> {"['had']-['some', 'milk']-[]-3": 3,
     "['walked']-['home']-[]-1": 1,
     "[]-['she']-['had', 'some', 'milk']-2": 2,
     "[]-['the', 'cat']-['walked', 'home']-0": 0}
print(phrases.norm_vectors.dot(phrases.norm_vectors.T))
>>> array([[1.        , 0.84016205, 0.39185037, 0.46664165],
           [0.84016205, 1.        , 0.23170359, 0.26717227],
           [0.39185037, 0.23170359, 1.        , 0.71929454],
           [0.46664165, 0.26717227, 0.71929454, 1.        ]])
r.save("phrases.vec")

```

The same applies to creating the concept vectors, except these are dicts mapping from a name to a list of tokenized descriptions.
These can then be composed using the code in `preprocessing.concept_vectors`

```python
from conch.preprocessing.concept_vectors import create_concepts

concepts = {"dog": ["an animal with four legs",
                    "a canine"],
            "cat": ["an animal with four legs and a tail"]}
r = Reach.load("my_embeddings.vec")

concept_vectors = create_concepts(concepts, r)

# concept_vectors is an embedding space containing concept vectors.
r.save("concepts.vec")
```

These can then be compared to your phrase vectors to infer things about them.
