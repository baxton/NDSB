
import os
import sys
import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import ctypes


ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\ann\ann.dll")



def plotit(X, Y, c='b', m='x', clear=True, show=True, subplot=None):
    if clear:
        plt.clf()
    if None != subplot:
        plt.subplot(*subplot)
    plt.scatter(X, Y, c=c, marker=m)
    if show:
        plt.show()


def gen_data(N, disp=False, subplot=(1,1,1)):

    np.random.seed()

    CX = np.array([0, 10, 8])
    CY = np.array([0, 10, -7])

    mul = 3.

    X1 = [d*mul + CX[0] for d in np.random.normal(size=N)]
    Y1 = [d*mul + CY[0] for d in np.random.normal(size=N)]

    X2 = [d*mul + CX[1] for d in np.random.normal(size=N/2)]
    Y2 = [d*mul + CY[1] for d in np.random.normal(size=N/2)]

    X3 = [d*mul + CX[2] for d in np.random.normal(size=9)]
    Y3 = [d*mul + CY[2] for d in np.random.normal(size=9)]


    if disp:
        plotit(CX, CY, c='b', m='x', clear=False,  show=False, subplot=subplot)
        plotit(X1, Y1, c='r', m='o', clear=False, show=False, subplot=subplot)
        plotit(X2, Y2, c='y', m='o', clear=False, show=False, subplot=subplot)
        plotit(X3, Y3, c='g', m='o', clear=False, show=False,  subplot=subplot)

    return sp.array(zip(X1, Y1)), sp.array(zip(X2, Y2)), sp.array(zip(X3, Y3))




def train(ds1, ds2, ds3):
    X = sp.concatenate((ds1, ds2, ds3), axis=0)
    #Y = sp.concatenate(( [[1,0,0]]*ds1.shape[0], [[0,1,0]]*ds2.shape[0], [[0,0,1]]*ds3.shape[0] ))
    Y = sp.concatenate(( [0]*ds1.shape[0], [1]*ds2.shape[0], [2]*ds3.shape[0] ))

    ann = RandomForestClassifier()
    ann.fit(X, Y)

    return ann

def train_ann(ds1, ds2, ds3):
    X = sp.concatenate((ds1, ds2, ds3), axis=0).astype(np.float32)
    Y = sp.concatenate(( [[1,0,0]]*ds1.shape[0], [[0,1,0]]*ds2.shape[0], [[0,0,1]]*ds3.shape[0] )).astype(np.float32)

    ann = ANN_DLL.ann_create()
    ANN_DLL.ann_fit(ctypes.c_void_p(ann), X.ctypes.data, Y.ctypes.data, ctypes.c_int(X.shape[0]), ctypes.c_float(.4), ctypes.c_float(15), ctypes.c_int(100))

    return ann




def test(rf, ann):
    plt.clf()
    ds1, ds2, ds3 = gen_data(200, disp=True, subplot=(3,1,1))



    ## sklearn RF
    for x in ds1:
        p = rf.predict(x)
        c = 'r' if p == 0 else 'y' if p == 1 else 'g'
        plotit(ds1[:,0], ds1[:,1], c=c, clear=False, show=False, subplot=(3,1,2))

    for x in ds2:
        p = rf.predict(x)
        c = 'r' if p == 0 else 'y' if p == 1 else 'g'
        plotit(ds2[:,0], ds2[:,1], c=c, clear=False, show=False, subplot=(3,1,2))

    for x in ds3:
        p = rf.predict(x)
        c = 'r' if p == 0 else 'y' if p == 1 else 'g'
        plotit(ds3[:,0], ds3[:,1], c=c, clear=False, show=False, subplot=(3,1,2))


    ## ANN
    for x in ds1:
        predictions = np.array([0] * 3, dtype=np.float32)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, predictions.ctypes.data, ctypes.c_int(1))
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'y' if p == 1 else 'g'
        plotit(ds1[:,0], ds1[:,1], c=c, clear=False, show=False, subplot=(3,1,3))

    for x in ds2:
        predictions = np.array([0] * 3, dtype=np.float32)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, predictions.ctypes.data, ctypes.c_int(1))
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'y' if p == 1 else 'g'
        plotit(ds2[:,0], ds2[:,1], c=c, clear=False, show=False, subplot=(3,1,3))

    for x in ds3:
        predictions = np.array([0] * 3, dtype=np.float32)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, predictions.ctypes.data, ctypes.c_int(1))
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'y' if p == 1 else 'g'
        plotit(ds3[:,0], ds3[:,1], c=c, clear=False, show=False, subplot=(3,1,3))


    ##
    plt.show()


def main():
    sp.random.seed()
    ds1, ds2, ds3 = gen_data(200, disp=False)
    rf = train(ds1, ds2, ds3)
    ann = train_ann(ds1, ds2, ds3)
    test(rf, ann)




if __name__ == '__main__':
    main()
