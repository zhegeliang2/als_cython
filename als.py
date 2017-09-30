#coding:utf-8
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time
import sys

def load_matrix(filename):
    #获取num_users和num_items
    num_users = 0 
    num_items = 0 
    user2id = {}
    item2id = {}
    scores = []
    for line in open(filename, "r"):
        line = line.strip()
        user, item, count = line.strip().split('\t')
        user = int(user)
        if user not in user2id:
            num_users += 1
            user2id[user] = num_users
        item = int(item)
        if item not in item2id:
            num_items += 1
            item2id[item] = num_items
        count = float(count)
        scores.append([user, item, count])
    t0 = time.time()
    print "num_user:%s, num_item:%s, score len:%s" %(num_users, num_items, len(scores))
    #counts = np.zeros((num_items, num_users))
    total = 0.0
    num_zeros = num_users * num_items
    '''如果要对一个列表或者数组既要遍历索引又要遍历元素时，可以用enumerate，当传入参数为文件时，索引为
    行号，元素对应的一行内容'''
    row_index = []
    col_index = []
    data =[]
    for i, line in enumerate(scores): 
        user, item, count = line
        if count != 0:
            data.append(count)
            row_index.append(user2id[user]-1)
            col_index.append(item2id[item]-1)
            #counts[item2id[item]-1, user2id[user]-1] = count
            total += count
            num_zeros -= 1
        if i % 10000 == 0:
            print 'loaded %i counts...' % i
    #数据导入完毕后计算稀疏矩阵中零元素个数和非零元素个数的比例，记为alpha
    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    #counts *= alpha
    #counts[counts>0] += 1
    #用CompressedSparse Row Format将稀疏矩阵压缩
    counts = sparse.csr_matrix((data, (row_index, col_index)), shape=(num_users, num_items))
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1 - t0)
    return item2id, counts


def alternating_least_squares(Cui, factors=10, regularization=0.01, iterations=10):
    users, items = Cui.shape

    np.random.seed(1)
    X = np.random.rand(users, factors) * 0.01
    Y = np.random.rand(items, factors) * 0.01

    Ciu = Cui.T.tocsr()
    for iteration in range(iterations):
        least_squares(Cui, X, Y, regularization)
        least_squares(Ciu, Y, X, regularization)

    return X, Y
def save_item(filename, item_vectors, item2id):
    out_f = open(filename, "w")
    for item in item2id:                                                                      
        id = item2id[item]
        line = item_vectors[id-1]
        out_f.write(str(item) + "\t" + "\t".join(map(str, line))+"\n")
    out_f.close()
def least_squares(ratings, X, Y, regularization):
    users, factors = X.shape
    YtY = Y.T.dot(Y)
    lambaI = np.eye(YtY.shape[0])*regularization

    for u in range(users):
        #x n*k, rating[u] 1*n
        X[u, :] = np.linalg.solve((YtY + lambaI), ratings[u, :].dot(Y).reshape(-1)) 
        #X[u, :] = np.linalg.solve((YtY + lambaI), ratings[u, :].dot(Y)) 
def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]
def train(filename):
    item2id, matrix= load_matrix(filename)
    X, Y = alternating_least_squares(matrix) 
    save_item("./song_vec.txt", Y, item2id) 
    print "ok"
def train_test_split(ratings):
    test = sparse.csr_matrix(ratings.shape)
    test_data = []
    test_row = []
    test_col = []
    for user in xrange(ratings.shape[0]):
        if len(ratings[user,:].nonzero()[0]) >=10:
            test_ratings = np.random.choice(ratings[user,:].nonzero()[1],
                    size=5,
                    replace=False)
            tmp_row = np.array([user]*len(test_ratings))
            test_data.extend(np.asarray(ratings[(tmp_row, test_ratings)]).reshape(-1).tolist())
            test_row.extend(tmp_row.tolist())
            test_col.extend(test_ratings.tolist())
    test = sparse.csr_matrix((test_data, (test_row, test_col)), shape=ratings.shape)
    train = ratings-test

    return train,test

def rmse(a, b):
    return np.sqrt(((a-b)**2).mean())
def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    #rmse
    return rmse(pred, actual)
def predict_all(X, Y):
    predictions = np.zeros((X.shape[0], Y.shape[0]))

    for u in xrange(X.shape[0]):
        for i in xrange(Y.shape[0]):
            predictions[u, i] = X[u, :].dot(Y[i, :].T)
    return predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage:input"
        sys.exit(1) 
    #train(sys.argv[1])
    item2id, ratings= load_matrix(sys.argv[1])
    train, test = train_test_split(ratings)
    print "split succ"
    X, Y = alternating_least_squares(train)
    print "train succ"
    preditions = predict_all(X, Y)
    train_mse = get_mse(preditions, train)
    test_mse = get_mse(preditions, test)
    print "train mse:" + str(train_mse)
    print "test mse:" + str(test_mse)
    print "succ"
