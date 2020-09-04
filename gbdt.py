from sklearn import linear_model
import numpy as np
import lightgbm as lgm
import time
from model import Model
from config import CONFIG
import logging


class GBDT(Model):
    def __init__(self):
        self.num_leaves = 100
        self.num_rounds = 100
        self.query_size = 100000

    def train(self, X, Y):
        #预处理
        for idx,k in enumerate(X):
            X[idx]=[int(k + "0" * (CONFIG["KEYLEN"] - len(k)), 16)]
        inputs=np.array(X)
        labels=np.array(Y)
        time_start = time.time()

        # STEP1: LR
        self.LR = linear_model.LinearRegression()
        self.LR.fit(inputs, labels)
        preds = self.LR.predict(inputs)

        logging.info("LR loss:"+str(max(abs(labels - preds))))

        # STEP2: GDBT
        train_data = lgm.Dataset(inputs, label=labels - preds)
        param = {'num_leaves': self.num_leaves, 'objective': 'mse', 'verbose': 0, 'device': 'gpu'}
        self.bst = lgm.train(param, train_data, self.num_rounds)
        preds2 = self.bst.predict(inputs)

        time_stop = time.time()
        logging.info(str(time_stop - time_start)+ 's')

        logging.info("final loss:"+str(max(abs(labels - preds - preds2))))

    def query(self, k):
        k=np.array([[int(k + "0" * (CONFIG["KEYLEN"] - len(k)), 16)]])
        return self.LR.predict(k) + self.bst.predict(k)
