"""Create text files for Cubner."""
import os
from glob import iglob


def read_cubner(pathtofile):
    """Read cubner output format."""
    words, tags = [], []
    count = 0
    index = 0

    for line in open(pathtofile):

        line = line.strip()

        if line:
            try:
                w, np, t = line.split("\t")
            except ValueError:
                count += 1
                continue
            words.append(w)
            tags.append(t)
            index += 1

    return words, tags


def write_cubner(data, pathtofile):
    """Write cubner output format."""
    f = open(pathtofile, 'w')
    for txt, bio in list(zip(*sorted(data.items())))[1]:
        for w, tag in zip(txt, bio):
            w = w.strip()
            f.write("{}\t{}\n".format(w, tag))
        f.write("\n")


def write_file(filename, paths):
    """Write all documents to a single file."""
    f = open(filename, 'w')
    for path in sorted(paths):
        for line in open(path):
            # Because not all lines end with newlines.
            f.write("{} ".format(line.strip()))
        f.write("\n")


if __name__ == "__main__":

    base = ""
    beth = iglob(os.path.join(base, "beth/*.txt"))
    partners = iglob(os.path.join(base, "partners/*.txt"))
    test = iglob(os.path.join(base, "test/*.txt"))

    write_file("data/beth_all.txt", beth)
    write_file("data/partners_all.txt", partners)
    write_file("data/test_all.txt", test)
