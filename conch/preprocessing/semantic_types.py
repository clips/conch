"""Create Semantic type identifiers."""
import json


if __name__ == "__main__":

    stys = {x["_id"]: x["concepts"] for x in json.load(open("data/stys.json"))}

    test = ['T060', 'T059', 'T034']
    treatment = ['T061', 'T200']
    problem = ['T020', 'T190', 'T049',
               'T019', 'T047', 'T050',
               'T033', 'T037', 'T048',
               'T191', 'T046', 'T184']

    dicto = {}

    for sty, concepts in stys.items():

        label = "np"

        if sty in problem:
            label = "problem"
        elif sty in test:
            label = "test"
        elif sty in treatment:
            label = "treatment"

        dicto.update({k: label for k in concepts})
