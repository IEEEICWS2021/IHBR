import pandas as pd
import numpy as np
import scipy.sparse as sp


def load_dataset():
    train_data_path = './data/user_bundle_train.txt'
    test_data_path = './data/test_negative.txt'
    user_bundle_path = './data/user_bundle.txt'
    user_item_path = './data/user_item.txt'
    bundle_item_path = './data/bundle_item.txt'

    train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'bundle'])
    user_bundle = pd.read_csv(user_bundle_path, sep='\t', header=None, names=['user', 'bundle'])
    user_item = pd.read_csv(user_item_path, sep='\t', header=None,names=['user', 'item'])
    bundle_item = pd.read_csv(bundle_item_path,sep='\t',header=None, names=['bundle', 'item'])

    user_num = 8039
    item_num = 32770
    bundle_num = 4771

    train_data = train_data.values.tolist()
    user_bundle_data = user_bundle.values.tolist()

    user_item_data = user_item.values.tolist()
    bundle_item_data = bundle_item.values.tolist()

    user_bundle_mat = sp.dok_matrix((user_num, bundle_num), dtype=np.float32)
    for x in user_bundle_data:
        user_bundle_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_data_path, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.strip().split('\t')
            user = eval(arr[0])[0]
            bundle = eval(arr[0])[1]
            test_data.append([user,bundle])
            for b in arr[1:]:
                test_data.append([user,eval(b)])
            line = fd.readline() 

    return train_data,test_data,user_bundle_data,user_item_data,bundle_item_data,item_num,user_num,bundle_num,user_bundle_mat

