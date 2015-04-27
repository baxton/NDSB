import os
import sys
import numpy as np
import scipy as sp
import scipy.spatial

visited = [
(1.6604389945673137e-09, 0.00083021949729378264, 1469053600.5853164, 'submission_2nd_1407734.874521_731.txt') ,
(1.7836072155940231e-09, 0.0008918036078084945, 1443824737.9825456, 'submission_2nd_1380408.397997_535.txt') ,
]

visited_files = [r[3] for r in visited]

path = "C:\\Temp\\test_python\\RRP\\"

best_name = "C:\\Temp\\test_python\\RRP\\best_1686914.txt"

files = [f for f in os.listdir(path) if "submission" in f and f not in visited_files]

def main():

    best_data = sp.loadtxt(best_name, delimiter=",", skiprows=1)

    result = []

    N = len(files)
    cnt = 0
    sums = np.zeros((100000,))

    for fn in files:
        cnt += 1
        print "processing", cnt, "out of", N

        data = sp.loadtxt(path + fn, delimiter=",", skiprows=1)

        cos = scipy.spatial.distance.cosine(data[:,1], best_data[:,1])
        length = np.sqrt( (data[:,1]**2).sum() )

        a = 1. / 1000000. * cos
        b = 100000000000. / length
        f = 2. * (a * b) / (a + b)

        result.append((f, cos, length, fn))

    result.extend(visited)

    result.sort()
    for r in result:
        print r, ","



if __name__ == '__main__':
    main()
