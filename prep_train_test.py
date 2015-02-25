

import os
import sys
import numpy as np

sys.path.append("..\\")
import pathes






THRESHHOLD = 200
FOR_TRAIN = .7


test_ids = []
train_ids = []



def split_file(fname):
    ids = []
    with open(pathes.path_ids + fname, "r") as fin:
        for line in fin:
            line = line.strip()
            tokens = line.split(" ")
            ids.append(int(tokens[0]))
    ids = np.array(ids, dtype=int)


    lines_num = len(ids)
    if lines_num > THRESHHOLD:
        num = int(lines_num * FOR_TRAIN)
        indices = np.array([False] * ids.shape[0], dtype=bool)
        indices[np.random.choice(range(lines_num), num)] = True
        train_ids.extend(ids[indices])
        test_ids.extend(ids[~indices])

    else:
        idx = np.random.randint(0, lines_num)
        indices = ids == ids[idx]
        test_ids.extend(ids[indices])
        train_ids.extend(ids[~indices])



def process():

    for f in os.listdir(pathes.path_ids):
        print "# processing %s" % f
        split_file(f)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)

    with open(pathes.path_data + "train_ids.txt", "w+") as fout:
        for v in train_ids:
            fout.write("%d%s" % (v, os.linesep))

    with open(pathes.path_data + "valid_ids.txt", "w+") as fout:
        for v in test_ids:
            fout.write("%d%s" % (v, os.linesep))



def main():
    process()

if __name__ == '__main__':
    main()
