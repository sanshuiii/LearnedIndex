import numpy as np
import math
from dataFS import DataFS
from sklearn import linear_model
from BTrees.IIBTree import IIBTree
from config import CONFIG

SAMPLE_FREQ=6

def cul_time(k):
    k = k[:16]
    ret = 0
    ret += int(k[8:10]) * 1000000
    ret += int(k[10:16])
    return ret

def extract_timestamp(k):
    k = k[:10]
    ret = 0
    if int(k[:2]) == 9:
        ret += 60 * 60 * 24 * 31
    ret += int(k[2:4]) * 60 * 60 * 24
    ret += int(k[4:6]) * 60 * 60
    ret += int(k[6:8]) * 60
    ret += int(k[8:10])
    return ret

class Linear():
    def __init__(self):
        self.w=[]
        self.b=[]
        self.GC=IIBTree()
        dataFS=DataFS()
        self.bias=int(extract_timestamp(dataFS.first_key)/SAMPLE_FREQ)

    def group(self,k):
        ret = extract_timestamp(k)
        return math.floor(ret / SAMPLE_FREQ) - self.bias

    def Cul_GC(self,inputx):
        G = []
        C = []
        for i in range(len(inputx)):
            G.append(self.group(inputx[i]))
            C.append(cul_time(inputx[i]))
        return G, C

    def train(self, X, Y):
        x = []
        y = []
        G,C= self.Cul_GC(X)
        pre_ret = G[0]
        pos = 0
        for i in range(len(X)):
            offset = Y[i]
            ret = G[i]
            if ret != pre_ret:
                self.train_sub_linear(x, y)
                for j in range(ret - pre_ret - 1):
                    self.w.append(0.0)
                    self.b.append(0)
                x = []
                y = []
                pre_ret = ret
                pos = pos + 1
            x.append(C[i])
            y.append(offset)
        self.train_sub_linear(x, y)

    def train_sub_linear(self, x, y):
        data = np.array(x).reshape(-1, 1)
        label = np.array(y).reshape(-1, 1)
        LR = linear_model.LinearRegression()
        LR.fit(data, label)
        self.w.append(LR.coef_)
        self.b.append(LR.intercept_)
        pre_y = LR.predict(data)
        errs=abs(pre_y-y)
        for idx,err in enumerate(errs):
            if err[0]>CONFIG["BLOCKSIZE"]:
                self.GC[int(x[idx])]=int(y[idx])


    def query(self, k):
        index = int(self.group(k))
        if index in self.GC.keys():
            return self.GC[index]
        offset = self.w[index] * cul_time(k) + self.b[index]
        return int(offset[0][0])