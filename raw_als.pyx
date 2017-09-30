#coding:utf-8
import numpy as np
cimport numpy as np
import pandas as pd
from libc.stdio cimport printf 
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from cython.parallel cimport prange
import time
import sys
import cython
from cython cimport floating
from cython import boundscheck
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas
from libc.stdio cimport printf 
import pandas as pd

cdef inline void posv(char * u, int * n, int * nrhs, floating * a, int * lda, floating * b,                                  int * ldb, int * info) nogil:
    if floating is double:
        cython_lapack.dposv(u, n, nrhs, a, lda, b, ldb, info)
    else:
        cython_lapack.sposv(u, n, nrhs, a, lda, b, ldb, info)
cdef inline floating dot(int *n, floating *sx, int *incx, floating *sy, int *incy) nogil:
    if floating is double:
        return cython_blas.ddot(n, sx, incx, sy, incy)
    else:
        return cython_blas.sdot(n, sx, incx, sy, incy)
def load_matrix(filename):
    #获取num_users和num_items
    num_users = 0 
    num_items = 0 
    user2id = {}
    item2id = {}
    scores = []
    t0 = time.time()
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


class ExplicitMF():
    def __init__(self, ratings, factors=10, reg=0.01, verbose=True):
        self.Cui= ratings
        self.Ciu = self.Cui.T.tocsr()
        self.users, self.items = ratings.shape
        self.factors = factors
        self.reg = reg
        self._v = verbose
    def train(self, n_iter = 10):
        np.random.seed(1)
        self.X = np.random.rand(self.users, self.factors) * 0.01
        self.Y = np.random.rand(self.items, self.factors) * 0.01
        self.partial_train(n_iter)

    def partial_train(self, n_iter):
        for iteration in range(n_iter):
            least_squares(self.Cui, self.X, self.Y, self.reg)
            least_squares(self.Ciu, self.Y, self.X, self.reg)
    def mse(self, data):
        return predict_all(self.X, self.Y, data)
    def eval(self, iter_array, test):
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)
            self.train_mse += [self.mse(self.Cui)]
            self.test_mse += [self.mse(test)]
            iter_diff = n_iter 
            if self._v:
                print 'Iteration: {}'.format(n_iter)
                print 'Train mse: ' + str(self.train_mse[-1])
                print 'Test mse: ' + str(self.test_mse[-1])
def save(filename, item_vectors):
    out_f = open(filename, "w")
    for item,line in enumerate(item_vectors):                                                      
        out_f.write(str(item) + "\t" + "\t".join(map(str, line))+"\n")
    out_f.close()

def save_item(filename, item_vectors, item2id):
    out_f = open(filename, "w")
    for item in item2id:                                                                      
        id = item2id[item]
        line = item_vectors[id-1]
        out_f.write(str(item) + "\t" + "\t".join(map(str, line))+"\n")
    out_f.close()
@cython.boundscheck(False)
def least_squares(ratings, floating[:, ::1] X, floating[:, ::1] Y, double regularization):
    dtype = np.float64 if floating is double else np.float32
    cdef int users=X.shape[0], factors = X.shape[1]
    YtY = np.dot(np.transpose(Y), Y)
    cdef floating * A
    cdef floating *b
    A = <floating *> malloc(sizeof(floating)*factors*factors)
    b = <floating *> malloc(sizeof(floating)*factors)
    cdef floating[:, :] initialA = YtY + regularization*np.eye(factors, dtype = dtype)
    cdef int i, j,index,one=1, err,u
    cdef int[:] indptr = ratings.indptr, indices = ratings.indices
    cdef double[:] data = ratings.data
    cdef double sum = 0.0,confidence

    for u in prange(users, nogil=True):
        #A is fixed
        memcpy(A, &initialA[0, 0],sizeof(floating)*factors*factors)
        #b ratings[u]*Y 1*n n*k
        for j in range(Y.shape[1]):
            b[j] = 0.0
            for index in range(indptr[u], indptr[u + 1]): 
                #i是列号
                i = indices[index]
                confidence = data[index]
                b[j] += confidence*Y[i, j] 
        posv("U", &factors, &one, A, &factors, b, &factors, &err)
        memcpy(&X[u, 0], b, sizeof(floating) * factors)
    free(A)
    free(b)
def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]

def map_ids(row, mapper):
    return mapper[row]
#load file user pandas
def load_data(filename):
    names = ['uid', 'mid', 'rating']
    df = pd.read_csv(filename, sep="\t", names=names)
    n_users = df.uid.unique().shape[0]
    n_items = df.mid.unique().shape[0]
    print "num of users:%s, num of items:%s" %(n_users, n_items)

    mid_to_idx = {}
    for (idx, mid) in enumerate(df.mid.unique().tolist()):
        mid_to_idx[mid] = idx
    uid_to_idx = {}
    for (idx, uid) in enumerate(df.uid.unique().tolist()):
        uid_to_idx[uid] = idx
    I = df.uid.apply(map_ids, args=[uid_to_idx]).as_matrix()
    J = df.mid.apply(map_ids, args=[mid_to_idx]).as_matrix()
    V = np.ones(I.shape[0])
    likes = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
    likes = likes.tocsr()
    
    return mid_to_idx, likes

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

#针对数据集合中的非0元素计算误差
def predict_all(floating[:, :] X, floating[:, :] Y, train):
    dtype = np.float64 if floating is double else np.float32
    cdef floating * A
    cdef floating *B
    #cdef floating[::1, :] new_Y = Y.copy_fortran()
    #A = <floating *> malloc(sizeof(floating)*factors)
    cdef int users=X.shape[0], factors = X.shape[1]
    cdef int[:] indptr = train.indptr, indices = train.indices
    cdef double[:] data = train.data
    cdef int one = 1,u, i, index,total=0
    cdef double sum = 0.0,confidence, score

    for u in range(users):
        #memcpy(A, &X[u, 0],sizeof(floating)*factors)
        for index in range(indptr[u], indptr[u + 1]): 
            #i是列号
            i = indices[index]
            #对于u和i计算score
            score = dot(&factors, &X[u, 0], &one, &Y[i, 0], &one)
            confidence = data[index]
            sum += (score-confidence)**2
            total += 1
    return np.sqrt(sum/total)
   
def train(filename):
    item2id, ratings= load_matrix(filename)
    train, test = train_test_split(ratings)
    print "split succ"
    latent_factors = [5, 10, 20, 40, 80]
    regularizations = [0.01, 0.1, 1., 10., 100.]
    regularizations.sort()
    iter_array = [1, 2, 5, 10, 25]

    best_params = {}
    best_params['n_factors'] = latent_factors[0]
    best_params['reg'] = regularizations[0]
    best_params['n_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None

    for fact in latent_factors:
        print 'Factors: {}'.format(fact)
        for reg in regularizations:
            print 'Regularization: {}'.format(reg)
            MF_ALS = ExplicitMF(train, factors=fact, \
                                reg=reg)
            MF_ALS.eval(iter_array, test)
            min_idx = np.argmin(MF_ALS.test_mse)
            if MF_ALS.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = MF_ALS.train_mse[min_idx]
                best_params['test_mse'] = MF_ALS.test_mse[min_idx]
                best_params['model'] = MF_ALS
                print 'New optimal hyperparameters'
                print pd.Series(best_params)
    #X, Y = alternating_least_squares(train)
    #print "train succ"
    #train_mse = predict_all(X, Y, train)
    #test_mse = predict_all(X, Y, test)
    #print "train mse:" + str(train_mse)
    #print "test mse:" + str(test_mse)

#def evaluation(filename):
#    latent_factors = [5, 10, 20, 40, 80]
#    regularizations = [0.01, 0.1, 1., 10.,]
#    regularizations.sort()
#    iter_array = [5, 10, 25, 50]
#
#    best_params = {}
#    best_params['n_factors'] = latent_factors[0]
#    best_params['reg'] = regularizations[0]
#    best_params['n_iter'] = 0
#    best_params['train_mse'] = np.inf
#    best_params['test_mse'] = np.inf
#    best_params['model'] = None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage:input"
        sys.exit(1) 
    train(sys.argv[1])
