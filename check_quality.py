import os
import sys
import numpy as np
import scipy as sp
import scipy.spatial




def stat(data1, data2):
    tmp = data1[:,1] - data2[:,1]
    tmp = tmp * tmp
    tmp = np.sum(tmp) / tmp.shape[0]
    mse = np.sqrt(tmp)

    dist = scipy.spatial.distance.euclidean(data1[:,1], data2[:,1])
    cos_dist = scipy.spatial.distance.cosine(data1[:,1], data2[:,1])

    std1 = sp.std(data1)
    std2 = sp.std(data2)

    mean1 = sp.mean(data1)
    mean2 = sp.mean(data2)

    tmp = (data1 - mean1) * (data2 - mean2)
    cor = tmp.sum() / np.sqrt( ((data1 - mean1)**2).sum() *  ((data2 - mean2)**2).sum())

    len1 = np.sqrt( (data1**2).sum() )
    len2 = np.sqrt( (data2**2).sum() )

    print "=== STAT"
    print "    MSE", mse
    print "    DIST", dist
    print "    COS", cos_dist
    print "    STD", std1, "vs", std2, "(", (std1 - std2), ")"
    print "    MEAN", mean1, "vs", mean2, "(", (mean1 - mean2), ")"
    print "    CORR", cor
    print "    LEN", len1, "vs", len2



def main():
    best_name = "C:\\Temp\\test_python\\RRP\\best_1686914.txt"
    bad_name = "C:\\Temp\\test_python\\RRP\\bad_1700824.txt"

    best_data = sp.loadtxt(best_name, delimiter=",", skiprows=1)
    bad_data = sp.loadtxt(bad_name, delimiter=",", skiprows=1)

    if 1 < len(sys.argv):
        tmp = sys.argv[1]
    else:
        tmp = "submission_AUG_a.02_l20_i5_INF_51x33_-9_90.txt"
    fname = "C:\\Temp\\test_python\\RRP\\%s" % tmp
    data = sp.loadtxt(fname, delimiter=",", skiprows=1)

    stat(data, best_data)
    stat(data, bad_data)



if __name__ == '__main__':
    main()
