from model import Model
from BTrees.OIBTree import OIBTree
from BTrees.IIBTree import IIBTree
import time
import logging

class Btree(Model):
    def __init__(self):
        self.model = OIBTree()

    def train(self, X, Y):
        t1 = time.time()
        for x, y in zip(X, Y):
            self.model[x] = y
        t2 = time.time()
        logging.info(str(t2 - t1))

    def query(self, key):
        return self.model[key]