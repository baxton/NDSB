

import os
import sys
import numpy as np
import scipy as sp
from array import array
from PIL import Image

sys.path.append("..\\")
import pathes


path_out_test = "C:\\Temp\\kaggle\\NDSB\\data\\test_ann1\\"
path_out = "C:\\Temp\\kaggle\\NDSB\\data\\train_ann1\\"


# from initial run:
# Max rows 428 Max cols 424


MAX_ROWS = 100
MAX_COLS = 100

SCALE = .2336


cls_id_map = None



def get_classes():
    res = {}

    with open(pathes.path_data + "classes.txt", "r") as fin:
        for line in fin:
            line = line.strip()
            tokens = line.split(" ")
            id = int(tokens[0])

            res[tokens[1]] = id
    return res



def vectorize_test(fname):

    fname_in = pathes.path_test + fname
    fname_out = path_out_test + fname + ".b"

    img = Image.open(fname_in)

    rows = int(img.size[1] * SCALE)
    cols = int(img.size[0] * SCALE)

    rimg = img.resize((cols, rows))     # reversed orde of dimentions
    a = np.asarray(rimg)

    vec = array('f', [1.] * (MAX_ROWS * MAX_COLS))    # float

    idx = 1
    for r in range(rows):
        for c in range(cols):
            vec[idx] = float(a[r,c]) / 255.
            idx += 1

    fout = open(fname_out, "wb+")
    vec.tofile(fout)
    fout.flush()
    fout.close()




def vectorize(id, folder, fname):

    if not os.path.exists(path_out + folder):
        os.mkdir(path_out + folder)

    fname_in = pathes.path_train + folder + "\\" + fname
    fname_out = path_out + folder + "\\" + fname + ".b"

    img = Image.open(fname_in)

    rows = int(img.size[1] * SCALE)
    cols = int(img.size[0] * SCALE)

    rimg = img.resize((cols, rows))     # reversed orde of dimentions
    a = np.asarray(rimg)

    cls = cls_id_map[folder]

    vec = array('f', [1.] * (1 + MAX_ROWS * MAX_COLS))    # float

    vec[0] = cls

    idx = 1
    for r in range(rows):
        for c in range(cols):
            vec[idx] = float(a[r,c]) / 255.
            idx += 1

    fout = open(fname_out, "wb+")
    vec.tofile(fout)
    fout.flush()
    fout.close()





def process(fname):
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip()
            tokens = line.split(" ")
            path_parts = tokens[1].strip("\"").split("\\\\")

            id = int(tokens[0])
            folder = path_parts[0]
            file = path_parts[1]

            print "# processing: ", id, folder, file

            vectorize(id, folder, file)
            #break


def process_train():
    global cls_id_map
    cls_id_map = get_classes()
    files = [pathes.path_ids + f for f in os.listdir(pathes.path_ids)]
    for f in files:
        vectorize_test(f)

    print "# Max rows", MAX_ROWS, "Max cols", MAX_COLS




def process_test():
    files = [f for f in os.listdir(pathes.path_test)]

##    import matplotlib.image as Image
##    max_rows = 0
##    max_cols = 0

    for f in files:
        vectorize_test(f)
##        img = Image.imread(pathes.path_test + f)
##        if max_rows < img.shape[0]:
##            max_rows = img.shape[0]
##        if max_cols < img.shape[1]:
##            max_cols = img.shape[1]

    ##print "# Max rows", max_rows, "Max cols", max_cols



def main():
    #process_train()
    process_test()


if __name__ == '__main__':
    main()
