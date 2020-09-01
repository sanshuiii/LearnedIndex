import lightgbm as lgb
from BasicFS import DataProvider
import numpy as np
from sklearn import linear_model


if __name__ == '__main__':
    data, label = DataProvider('sorted_demo_data').gen_test_data()
    #data, label = z_score(data, label)
    data = np.array([[x] for x in data])
    label = np.array(label)
    LR = linear_model.LinearRegression()
    LR.fit(data, label)
    y1 = LR.predict(data)
    label2 = label - y1
    train_data = lgb.Dataset(data, label=label2)
    train_data.save_binary('train.bin')
    param = {'num_leaves': 15, 'objective': 'regression'}
    num_round = 15
    bst = lgb.train(param, train_data, num_round)
    bst.save_model('model.txt')
    ypred = bst.predict(data) + y1
    #_, ypred = de_score(data, ypred)
    err = ypred - label
    print(max(err))
