import lightgbm as lgb
from BasicFS import DataProvider
import numpy as np
from sklearn import linear_model
import pandas as pd
from gbdt_config import GBDT_CONFIG

def grid(num_round = 20,num_leaves=20):
    label2 = label - y1
    train_data = lgb.Dataset(data, label=label2)
    train_data.save_binary('train.bin')
    param = {'num_leaves': num_leaves, 'objective': 'regression'}
    bst = lgb.train(param, train_data, num_round)
    # bst.save_model('model.txt')
    ypred = bst.predict(data) + y1
    # _, ypred = de_score(data, ypred)
    err = ypred - label
    return max(err)


if __name__ == '__main__':
    filename=GBDT_CONFIG.get('filename')
    data, label = DataProvider('sorted_demo_data').gen_test_data()
    #data, label = z_score(data, label)
    data = np.array([[x] for x in data])
    label = np.array(label)
    LR = linear_model.LinearRegression()
    LR.fit(data, label)
    y1 = LR.predict(data)
    lbegin=GBDT_CONFIG.get('leaf_begin',10)
    lend=GBDT_CONFIG.get('leaf_end',)
    nbegin = GBDT_CONFIG.get('node_begin', 10)
    nend = GBDT_CONFIG.get('node_end', )
    df=pd.DataFrame(columns=list(range(nbegin,nend)),index=list(range(lbegin,lend)))
    for i in range(lbegin,lend):
        for j in range(nbegin,nend):
            df[i][j]=grid(i,j)
    df.to_csv('gbdt_search.csv')
