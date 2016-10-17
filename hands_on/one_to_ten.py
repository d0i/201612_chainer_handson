#!/usr/bin/env python

"""
very simple program to learn 1 to 10
"""

import chainer
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
import numpy as np
import pdb

class Model(chainer.ChainList):
    def __init__(self):
        self.e = L.EmbedID(10, 10)
        self.l = L.Linear(10, 1)
        chainer.ChainList.__init__(self, self.e, self.l)
    def __call__(self, v):
        return self.l(self.e(v))
    def lossfunc(self, v, t):
        l = abs(self(v)-t)
        return l

m = Model()

lr = 0.01
optimizer = optimizers.SGD(lr=lr)
optimizer.setup(m)

# learn
for j in range(100):
    for i in range(100):
        v = i % 10
        a = np.array([v], dtype=np.int32)
        optimizer.update(m.lossfunc, a, v)
    lr *= 0.96
    optimizer.lr = lr

    lsum = 0.0
    for i in range(10):
        a = np.array([i], dtype=np.int32)
        r = m(a).data[0]
        l = abs(a[0]-r[0])
        lsum += l
    print("learn:", lsum, lr)

# test
for i in range(10):
    a = np.array([i], dtype=np.int32)
    r = m(a).data[0]
    print(a[0], r[0], abs(a[0]-r[0]))
    print(m.e.W[i].data.dot(m.l.W.data.T)+m.l.b).data[0] # manual calculation

