# encoding=utf8
import argparse
import numpy as np
import math
import networkx as nx

import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from sparsesvd import sparsesvd

from sklearn import preprocessing
from scipy.special import iv
from scipy.fftpack import fft,ifft

from sklearn.utils.extmath import randomized_svd

import time
import random

min_value = 10e-100


class argParser(object):
    pass
args = argParser()

args.D = 128

def get_embedding_rand(matrix):
    t1 = time.time()
    l = matrix.shape[0]
    smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
    print('svd sparse', smat.data.shape[0]*1.0/l**2)
    U, Sigma, VT = randomized_svd(smat,
                              n_components=args.D,
                              n_iter=5,
                              random_state=None)
    U = U * np.sqrt(Sigma)
    U = preprocessing.normalize(U, "l2")
    # print('sparsesvd time', time.time()-t1)
    return U


def get_embedding(matrix):
    t1 = time.time()
    l = matrix.shape[0]
    smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
    print('svd sparse', smat.data.shape[0]*1.0/l**2)
    ut, s, vt = sparsesvd(smat, args.D)  # do SVD, asking for 100 factors
    emb_matrix = ut
    emb_matrix = emb_matrix * np.sqrt(s).reshape(args.D, 1)
    emb_matrix = emb_matrix.transpose()
    emb_matrix = preprocessing.normalize(emb_matrix, "l2")
    # print('sparsesvd time', time.time()-t1)
    return emb_matrix

def get_embedding_dense(matrix, d=args.D):
    t1 = time.time()
    l = matrix.shape[0]
    U, s, Vh = linalg.svd(matrix, full_matrices=False,  check_finite=False, overwrite_a=True)
    U = np.array(U)
    U = U[:, :d]
    s = s[:d]
    s = np.sqrt(s)
    U = U * s
    U = preprocessing.normalize(U, "l2")
    # print('densesvd time', time.time()-t1)
    return U

def pre_factorization(tran, mask, l1):
    t1 = time.time()

    C1 = preprocessing.normalize(tran, "l1")

#     l1 = 1
#     l2 = 1
#     neg = np.array(C1.sum(axis=0))[0]**l1 /node_number**l2
    neg = np.array(C1.sum(axis=0))[0]**l1
    neg = neg/neg.sum()
    neg = scipy.sparse.diags(neg, format="csr")
    neg = mask.dot(neg)
    l3 = 1
    neg = l3*neg + (1-l3)*C1
    # print("neg", time.time()-t1)

    C1.data[C1.data<=0] = 1
    neg.data[neg.data<=0] = 1

    C1.data = np.log(C1.data)
    neg.data = np.log(neg.data)

    C1 -= neg
    # F = get_embedding(C1)
    F = C1
    return F

def ChebyshevGaussian(A, a, order = 5, mu = 0.5, s = 0.5):
    # print('Chebyshev Series -----------------')
    t1 = time.time()

    if order == 1:
        return a
    node_number = A.shape[0]
    A = sp.eye(node_number) + A
    DA = preprocessing.normalize(A, norm='l1')
#     DAD = preprocessing.normalize(DAD.T, norm='l1').T
    L = sp.eye(node_number) - DA
    # print('Laplacian', time.time()-t1)

    M = L - mu* sp.eye(node_number)


    # print(abs(sp.linalg.eigs(sp.eye(node_number) - L, k=20, which='LM', return_eigenvectors=False)))
    # assert False
#     lmax = sp.linalg.eigs(L, k=1, which='LM', return_eigenvectors=False)[0].real
#     print('l max time', time.time()-t1)
#     L = 2.0/lmax *L - sp.eye(node_number)
#     print(lmax)

    Lx0 = a
    Lx1 = M.dot(a)
    Lx1 = 0.5* M.dot(Lx1) - a

    conv = iv(0,s)*Lx0
    conv -= 2*iv(1,s)*Lx1
    for i in range(2, order):
        Lx2 = M.dot(Lx1)
        Lx2 = (M.dot(Lx2) - 2*Lx1) - Lx0
#         Lx2 = 2*L.dot(Lx1) - Lx0
        if i%2 ==0:
            conv += 2*iv(i,s)*Lx2
        else:
            conv -= 2*iv(i,s)*Lx2
        Lx0 = Lx1
        Lx1 = Lx2
        del Lx2
        # print('sparsity',i, conv.data.shape[0]/node_number**2)
        # print('Bessell time',i, time.time()-t1)
#     return conv
#     print(np.sum(A.dot(a-math.pi*conv)<0), np.sum(A.dot(a-math.pi*conv)>0))
    return A.dot(a-conv)

def pros(matrix0):
    node_number = matrix0.shape[0]
    l1 = 0.75
    features_matrix = pre_factorization(matrix0, matrix0, l1)
    # features_matrix = get_embedding_rand(features_matrix)
    features_matrix = get_embedding(features_matrix)

    step = 10
    theta = 0.5
    mu = 0.1

    mm = ChebyshevGaussian(matrix0, features_matrix, order = step, mu = mu, s = theta)
    emb = get_embedding_dense(mm)

    return features_matrix




