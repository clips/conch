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

## usage

Conch requires the following to work.

* A set of word vectors
* A mapping from concepts to their descriptions or strings
* A parser

If you want to replicate the experiments in the paper, you will need access to the [i2b2-2010](https://www.i2b2.org/NLP/DataSets/Main.php) challenge corpus.

If you have access to the i2b2-2010 challenge corpus, please run all the preprocessing scripts in `conch.preprocessing` to extract noun phrases, and convert the gold standard data to `IOB` format. We currently offer a conversion script from `UIMA` `XML` format to `IOB` format. If you use another parser or chunker, you will have to write your own converter.

Concept representations are also created using a preprocessing script. The input to this script is a dictionary (we use a JSON file), with the UMLS CUIs as keys, and the descriptions as lists of strings.
