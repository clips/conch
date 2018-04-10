"""Create concept vectors."""
import numpy as np
import json

from tqdm import tqdm
from reach import Reach


def create_concepts(concepts,
                    embeddings,
                    include_np=True):
    """Create concepts by summing over descriptions in embedding spaces."""
    # Gold standard labels for concepts:
    sty = json.load(open("data/concept_label.json"))

    concept_names = []
    vectors = []

    concept_labels = []

    for name, descriptions in tqdm(list(concepts.items())):

        try:
            label = sty[name]
        except KeyError:
            continue

        if not include_np and label == "np":
            continue

        concept = []

        for idx, desc in enumerate(descriptions):

            try:
                desc = desc.lower().split()
                # desc = [x for x in desc if x not in STOP_WORDS]
                vec = embeddings.vectorize(desc, remove_oov=True)
                if not np.any(vec):
                    continue
                concept.append(np.mean(vec, axis=0))
            except ValueError:
                pass

        if not concept:
            continue

        concept_labels.append(label)
        name = "{0}_{1}".format(name, "_".join(descriptions[0].split()))
        concept_names.append(name)
        vectors.append(np.array(concept).mean(axis=0))

    r = Reach(np.array(vectors), concept_names)
    name2label = dict(zip(concept_names, concept_labels))

    return r, name2label


if __name__ == "__main__":

    path_to_embeddings = ""
    r_1 = Reach.load(path_to_embeddings, unk_word="UNK")

    concepts = json.load(open("data/all_concepts.json"))
    r, name2label = create_concepts(concepts, r_1, include_np=True)

    r.save_fast_format("data/concept_vectors")
    json.dump(name2label, open("data/names2label.json", 'w'))
