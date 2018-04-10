"""Extract sentence and NP chunks from documents parsed with ctakes."""
import json
import os

from collections import OrderedDict, Counter
from lxml import etree
from io import open
from glob import glob

NS = {'refsem': 'http:///org/apache/ctakes/typesystem/type/refsem.ecore',
      'cas': 'http:///uima/cas.ecore',
      'textspan': 'http:///org/apache/ctakes/typesystem/type/textspan.ecore',
      'syntax': 'http:///org/apache/ctakes/typesystem/type/syntax.ecore',
      'textsem': 'http:///org/apache/ctakes/typesystem/type/textsem.ecore'}


def get_sentences(text):
    """Get all sentences from a ctakes document."""
    sentences = OrderedDict()
    prev = 0

    for line in text.split("\n"):

        if not line:
            continue

        position = prev + len(line) + 1

        sentences[position] = (prev, position, text[prev:position-1], [])
        prev = position

    return sentences


def get_chunks(ns, root, text, sentences):
    """Extract all NP chunks from a text and assign them to sentences."""
    sents, bios = [], []

    chunk_annotations = root.findall('syntax:Chunk', ns)
    skipped = done = 0
    true = 0

    for chunk in chunk_annotations:

        chunk_type = chunk.get('chunkType')

        if chunk_type != "NP":
            continue

        true += 1
        # Compute overlap.
        begin = int(chunk.get('begin'))
        end = int(chunk.get('end'))

        tokens = text[begin:end]

        for k in sentences.keys():
            if begin < k:

                new_chunk = (begin, end, tokens, chunk_type)
                sentences[k][3].append(new_chunk)
                break

    for k, v in sentences.items():

        l_b, l_e, txt, chunks = v

        bio = ["O"] * len(txt.split())

        for b, e, w, _ in chunks:

            b -= l_b
            e -= l_b

            ch_len = len(w.split())
            start = len(txt[:b].split())

            for x in range(ch_len):

                try:
                    if x == 0:
                        bio[start + x] = "B-NP"
                    else:
                        bio[start + x] = "I-NP"
                except IndexError:
                    skipped += 1
                    break
            else:
                done += 1
        sents.extend(txt.split())
        bios.extend(bio)

    print(skipped, done, true, Counter(bios))

    return sents, bios


def process(paths):
    """
    Process a set of ctakes parsed documents.

    Parameters
    ==========
    paths : list of string
        A list of paths to the XML files being parsed.

    Returns
    =======
    chunks : dict
        A dict of tuples, where the key of the dictionary is the file-name,
        and the value is a tuple of lists. The first list in each tuple is the
        raw text, and the second list in each tuple is the BIO string of said
        text.

    """
    chunks = {}

    for path in paths:

        name = os.path.splitext(os.path.split(path)[-1])[0]
        root = etree.parse(open(path)).getroot()

        for cas in root.findall('cas:Sofa', NS):
            for attr in cas.attrib:
                if attr == 'sofaString':
                    content = cas.get(attr)

        sentences = get_sentences(content)
        chunks[name] = get_chunks(NS, root, content, sentences=sentences)

    return chunks


if __name__ == "__main__":

    base = ""
    g = glob(os.path.join(base, "beth/*.xml"))
    beth = process(g)
    g = glob(os.path.join(base, "partners/*.xml"))
    partners = process(g)

    json.dump(beth, open("data/beth_uima.json", 'w'))
    json.dump(partners, open("data/partners_uima.json", 'w'))
    beth.update(partners)
    json.dump(beth, open("data/train_uima.json", 'w'))

    g = glob(os.path.join(base, "test/*.xml"))
    result = process(g)
    json.dump(result, open("data/test_uima.json", 'w'))
