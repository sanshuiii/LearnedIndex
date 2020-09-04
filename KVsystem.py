import logging
from dataFS import DataFS
import config
import time
import sys
import inspect
import random
from linear import Linear
from btree import Btree
from gbdt import GBDT

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='KVsystem.log', level=logging.DEBUG, format=LOG_FORMAT)


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))
    return size

class KVsystem:
    def __init__(self):
        self.dataFS = DataFS()
        self.model = self.choose_model()

    def choose_model(self):
        if config.CONFIG['MODEL_TYPE'] == 'btree':
            return Btree()
        elif config.CONFIG["MODEL_TYPE"]=='gbdt':
            return GBDT()
        elif config.CONFIG["MODEL_TYPE"]=='linear':
            return Linear()

    def build(self):
        X, Y = self.dataFS.gen_train_data()
        self.model.train(X, Y)

    def query(self, key):
        return self.dataFS.query(key, self.model.query(key))

    def gen_keys(self):
        X, _ = self.dataFS.gen_train_data()
        return X

if __name__ == '__main__':
    kv = KVsystem()
    kv.build()
    X = kv.gen_keys()

    random.shuffle(X)
    t1 = time.time()
    for x in X:
        kv.query(x)
    t2 = time.time()
    print(t2 - t1)
