# -*- coding: utf8 -*-

import os
import random
import logging
import mxnet as mx
import numpy as np
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon import loss

from data import *
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y%m%d-%H:%M:%S',
    format='%(asctime)s:  %(message)s',
)

###################################################

def getBS(name):
    bSs = {
        'dffm31'    : 200,
        'dfcm12'    : 500,
        'dfcm13'    : 500,
    }

    if name in bSs:
        return bSs[name]
    return 2000

def getOpt(iters):
    opt = mx.optimizer.AdaGrad(
        wd=1e-3,
        learning_rate=1e-2,
    )

    return opt

def getSorN(name):
    models = (
        'dfr',
    )

    return name not in models

def getStart(name):
    dims = getFFMDim().values()

    models = {
        'dcn'   : lambda: MyDCN(dims, 90947, 8, 64, 3),
        'dfcm'  : lambda: MyDFCM(dims, 90947, 8, 64),       # 7370, 7406, XXXX
        'dfcm11': lambda: MyDFCM(dims, 90947, 16, 128),     # XXXX, 7431, XXXX
        'dfcm12': lambda: MyDFCM(dims, 90947, 32, 128),     # XXXX, 7482, XXXX
        'dfcm13': lambda: MyDFCM(dims, 90947, 32, 256),     # XXXX, 7485, XXXX

        'dfr'   : lambda: MyDFR(dims, 90947, 8, 64),
        'dfu'   : lambda: MyDFU(dims, 90947, 8, 64),        # 7269, XXXX, XXXX
        'dfm'   : lambda: MyDFM(dims, 90947, 8, 64),        # 7256, 7280, XXXX

        'dfm2'  : lambda: MyDFM2(dims, 90947, 8, 64),       # XXXX, 7369, XXXX
        'dfm21' : lambda: MyDFM2(dims, 90947, 16, 128),     # XXXX, 7410, XXXX
        'dfm22' : lambda: MyDFM2(dims, 90947, 32, 128),     # XXXX, 7445, XXXX
        'dfm23' : lambda: MyDFM2(dims, 90947, 32, 256),     # XXXX, 7450, XXXX
        'dfm24' : lambda: MyDFM2(dims, 90947, 64, 256),     # XXXX, 7468, XXXX
        'dfm25' : lambda: MyDFM2(dims, 90947, 128, 256),    # XXXX, 7468, XXXX
        'dfm26' : lambda: MyDFM2(dims, 90947, 256, 256),    # XXXX, 7475, XXXX

        'dfz'   : lambda: MyDIN(dims, 90947, 8, 64),        # XXXX, 7325, XXXX
        'dfz11' : lambda: MyDFZ(dims, 90947, 16, 128),      # XXXX, 7402, XXXX
        'dfz12' : lambda: MyDFZ(dims, 90947, 32, 128),      # XXXX, 7443, XXXX
        'dfz13' : lambda: MyDFZ(dims, 90947, 32, 256),      # XXXX, 7439, XXXX
        'dfz14' : lambda: MyDFZ(dims, 90947, 64, 256),      # XXXX, 7456, XXXX
        'dfz15' : lambda: MyDFZ(dims, 90947, 128, 256),     # XXXX, 7463, XXXX
        'dfz16' : lambda: MyDFZ(dims, 90947, 256, 256),     # XXXX, 7469, XXXX

        'din'   : lambda: MyDIN(dims, 90947, 8, 64),        # XXXX, 7325, XXXX
        'dfin'  : lambda: MyDFIN(dims, 90947, 8, 64),       # 7399, 7415, XXXX
        'dfcn'  : lambda: MyDFCN(dims, 90947, 8, 64),       # 7415, XXXX, XXXX

        'dffm'  : lambda: MyDFFM(dims, 90947, 8, 64),       # 7395, 7439, XXXX
        'dffm2' : lambda: MyDFFM2(dims, 90947, 8, 64),      # 7396, 7444, XXXX
        'dffm3' : lambda: MyDFFM3(dims, 90947, 8, 64),      # 7431, 7456, 7436
        'dffm31': lambda: MyDFFM3(dims, 90947, 16, 64),     # XXXX, 7462, XXXX
    }

    return models[name]()

def getLoss(lossKind):
    losses = {
        0: MyLoss,
        1: MyLoss2,
        2: MyLoss3,
    }
    return losses[lossKind]()

def getMetric(lossKind):
    metrics = {
        0: MyMetric,
        1: MyMetric2,
        2: MyMetric3,
    }
    return metrics[lossKind]()

def randomRange(start, end):
    r1 = r2 = 0
    while(r1 == r2):
        r1 = random.randint(start, end)
        r2 = random.randint(start, end)
    t1, t2 = min(r1,r2), max(r1,r2)
    return t1, t2

###################################################

class MyBA(nn.HybridSequential):
    def __init__(self, act='relu'):
        super(MyBA, self).__init__()

        with self.name_scope():
            self.add(nn.BatchNorm())
            self.add(MyAct(act))

class MyAct(nn.HybridBlock):
    def __init__(self, act='relu'):
        super(MyAct, self).__init__()
        self.act = act

    def hybrid_forward(self, F, x):
        if self.act == 'self': return x
        return F.Activation(x, self.act)

class MyIBA(nn.HybridSequential):
    def __init__(self, c, act='relu'):
        super(MyIBA, self).__init__()

        with self.name_scope():
            self.add(nn.Dense(c))
            self.add(MyBA(act))

class MyCBA(nn.HybridSequential):
    def __init__(self, c, act='relu'):
        super(MyCBA, self).__init__()

        with self.name_scope():
            self.add(nn.Conv1D(*c))
            self.add(MyBA(act))

class MyRes(nn.HybridBlock):
    def __init__(self, c1, c2):
        super(MyRes, self).__init__()

        with self.name_scope():
            self.opr = MyAct('relu')
            self.op1 = MyIBA(c1, 'self')
            self.op2 = MyIBA(c2, 'relu')

    def hybrid_forward(self, F, x):
        return self.op2(self.opr(self.op1(x)+x))

class MyRes2(nn.HybridSequential):
    def __init__(self, c1, c2, num):
        super(MyRes2, self).__init__()

        with self.name_scope():
            for i in xrange(num):
                self.add(MyRes(c1, c1))
            self.add(MyRes(c1, c2))

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

class MyC(nn.HybridBlock):
    def __init__(self, shape):
        super(MyC, self).__init__()

        with self.name_scope():
            self.b = self.params.get('b', shape=shape)
            self.w = self.params.get('w', shape=shape)

    def hybrid_forward(self, F, x0, x1, b, w):
        y = F.broadcast_add(F.dot(
            F.batch_dot(x0, x1, False, True), w), b)
        return y

class MyE(nn.HybridBlock):
    def __init__(self, inC, outC):
        super(MyE, self).__init__()

        with self.name_scope():
            self.inC = inC
            self.outC = outC

            self.w = self.params.get(
                'weight',
                shape=(inC-1, outC),
                allow_deferred_init=True
            )

    def hybrid_forward(self, F, x, w):
        zw = F.concat(
            F.zeros((1, self.outC)), w, dim=0)
        return F.Embedding(x, zw, self.inC, self.outC)

class MyEB(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyEB, self).__init__()

        self.d = dims
        self.e = MyE(inC, outC)

    def onDone(self, F, result):
        return result

    def hybrid_forward(self, F, x):
        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            result.append(F.mean(sliced, 1, True))

        return self.onDone(F, result)

class MyED(MyEB):
    def __init__(self, dims, inC, outC):
        super(MyED, self).__init__(dims, inC, outC)

    def onDone(self, F, result):
        return F.concat(*result, dim=1)

class MyER(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyER, self).__init__()

        self.d = dims
        self.e = MyE(inC, outC)

    def hybrid_forward(self, F, x):
        flag = autograd.is_recording()

        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size

            t1, t2 = start, end
            if flag and size > 5:
                t1, t2 = randomRange(start, end)

            sliced = F.slice_axis(e, 1, t1, t2)
            result.append(F.mean(sliced, 1, True))

        return F.concat(*result, dim=1)

class MyU(nn.HybridBlock):
    def __init__(self, inC, outC):
        super(MyU, self).__init__()

        with self.name_scope():
            self.inC = inC
            self.outC = outC

            self.w = self.params.get(
                'weight',
                shape=(inC-1, outC),
                allow_deferred_init=True,
                init=mx.init.Uniform(0.1),
            )

    def hybrid_forward(self, F, x, w):
        zw = F.concat(
            F.zeros((1, self.outC)), w, dim=0)
        return F.Embedding(x, zw, self.inC, self.outC)

class MyUD(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyUD, self).__init__()

        self.d = dims
        self.e = MyU(inC, outC)

    def hybrid_forward(self, F, x):
        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            result.append(F.mean(sliced, 1, True))
        return F.concat(*result, dim=1)

class MyZ(nn.HybridBlock):
    def __init__(self, inC, outC):
        super(MyZ, self).__init__()

        with self.name_scope():
            self.inC = inC
            self.outC = outC

            self.w = self.params.get(
                'weight',
                shape=(inC, outC),
                allow_deferred_init=True
            )

    def hybrid_forward(self, F, x, w):
        return F.Embedding(x, w, self.inC, self.outC)

class MyZB(nn.HybridBlock):
    def __init__(self, dims, inC, outC):
        super(MyZB, self).__init__()

        self.d = dims
        self.e = MyZ(inC, outC)

    def onDone(self, F, result):
        return result

    def hybrid_forward(self, F, x):
        e = self.e(x)
        result, start, end = [], 0, 0
        for i, size in enumerate(self.d):
            start, end = end, end + size
            sliced = F.slice_axis(e, 1, start, end)
            result.append(F.mean(sliced, 1, True))

        return self.onDone(F, result)

###################################################

class MyDCN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit, depth):
        super(MyDCN, self).__init__()

        with self.name_scope():
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC)

            self.depth = depth
            for i in xrange(depth):
                setattr(
                    self, 'MyC#%02d'%i,
                    MyC((len(dims)*outC, 1))
                )
                setattr(
                    self, 'MyBA#%02d'%i,
                    MyBA()
                )

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))
        deep = self.ip(F.flatten(e))

        y = e = F.expand_dims(F.flatten(e), -1)
        for i in xrange(self.depth):
            mc = getattr(self, 'MyC#%02d'%i)
            ba = getattr(self, 'MyBA#%02d'%i)

            y = ba(mc(e, y))
        cross = F.flatten(y)

        return self.ba(F.concat(deep, cross, dim=1))

class MyDFM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFM, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)
        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = 0.5*(order2_1-order2_2)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFM2(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFM2, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyEB(dims, inC, outC)

    def hybrid_forward(self, F, x):
        x = self.ed(x)
        y = F.concat(*x, dim=1)

        e = self.be(y)
        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in xrange(self.n):
            for k in xrange(i+1, self.n):
                order2.append(x[i]*x[k])
        order2 = F.flatten(F.concat(*order2, dim=1))

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFU(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFU, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyUD(dims, inC, outC)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)
        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = 0.5*(order2_1-order2_2)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFR(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFR, self).__init__()

        with self.name_scope():
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyER(dims, inC, outC)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)
        order2_1 = F.square(F.sum(e, axis=1))
        order2_2 = F.sum(F.square(e), axis=1)
        order2 = 0.5*(order2_1-order2_2)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFZ(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFZ, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyZB(dims, inC, outC)

    def hybrid_forward(self, F, x):
        x = self.ed(x)
        y = F.concat(*x, dim=1)

        e = self.be(y)
        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in xrange(self.n):
            for k in xrange(i+1, self.n):
                order2.append(x[i]*x[k])
        order2 = F.flatten(F.concat(*order2, dim=1))

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFCN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFCN, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC*n)

            for i in xrange(n):
                layer = nn.HybridSequential()
                layer.add(MyBA('relu'))
                layer.add(MyCBA((n,3,1,1), 'relu'))
                layer.add(MyCBA((n,3,1,1), 'relu'))
                layer.add(MyCBA((n,3,1,1), 'relu'))
                setattr(
                    self, 'MyLayer#%02d'%i, layer
                )

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        ss, es = [], F.split(e, self.n, 1)
        for s in es: ss.append(F.split(s,self.n,2))

        order2 = []
        for i in xrange(self.n):
            temp = []
            for k in xrange(self.n):
                temp.append(ss[i][k]*ss[k][i])
            temp = F.concat(*temp, dim=1)
            temp = F.swapaxes(temp, 1, 2)

            layer = getattr(self, 'MyLayer#%02d'%i)
            temp = F.swapaxes(layer(temp), 1, 2)
            order2.append(F.sum(temp, axis=1))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFCM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFCM, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyEB(dims, inC, outC)

    def hybrid_forward(self, F, x):
        es = self.ed(x)
        e = self.be(F.concat(*es, dim=1))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in xrange(self.n):
            for k in xrange(i+1, self.n):
                order2.append(F.batch_dot(
                    es[i], es[k], True, False))
        order2 = F.flatten(F.concat(*order2, dim=1))

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDIN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit, unit2=0):
        super(MyDIN, self).__init__()

        with self.name_scope():
            self.n = len(dims)
            self.ba = MyBA('relu')
            self.be = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyEB(dims, inC, outC)

            unit2 = unit2 or outC
            for i in xrange(self.n):
                for k in xrange(i+1, self.n):
                    setattr(
                        self, 'fc#%02d#%02d'%(i,k),
                        MyIBA(unit2, 'relu')
                    )

    def hybrid_forward(self, F, x):
        x = self.ed(x)
        y = F.concat(*x, dim=1)

        e = self.be(y)
        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        order2 = []
        for i in xrange(self.n):
            for k in xrange(i+1, self.n):
                xi, xk = x[i], x[k]
                fi = F.concat(xi, xi-xk, xk, dim=1)
                f = getattr(self, 'fc#%02d#%02d' % (i,k))

                order2.append(f(fi))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFIN(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFIN, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC*n)

            for i in xrange(n):
                for k in xrange(i+1, n):
                    setattr(
                        self, 'fc#%02d#%02d'%(i,k),
                        MyIBA(outC, 'relu')
                    )

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        ss, es = [], F.split(e, self.n, 1)
        for s in es: ss.append(F.split(s,self.n,2))

        order2 = []
        for i in xrange(self.n):
            temp = []
            for k in xrange(i+1, self.n):
                eik, eki = ss[i][k], ss[k][i]
                fi = F.concat(eik, eik-eki, eki, dim=1)
                f = getattr(self, 'fc#%02d#%02d' % (i,k))

                temp.append(f(fi))
            if len(temp) == 0: continue
            order2.append(F.concat(*temp, dim=1))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

class MyDFFM(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFFM, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ed = MyED(dims, inC, outC*n)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        ss, es = [], F.split(e, self.n, 1)
        for s in es:
            ss.append(F.split(s, self.n, 2))

        result = []
        for i in xrange(self.n):
            for k in xrange(i, self.n):
                result.append(ss[i][k]*ss[k][i])

        return F.concat(*result, dim=1)

class MyDFFM2(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFFM2, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ed = MyED(dims, inC, outC*n)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        ss, es = [], F.split(e, self.n, 1)
        for s in es:
            ss.append(F.split(s, self.n, 2))

        result = []
        for i in xrange(self.n):
            temp = []
            for k in xrange(self.n):
                temp.append(ss[i][k]*ss[k][i])
            temp = F.concat(*temp, dim=1)
            result.append(F.sum(temp, axis=1))

        return F.concat(*result, dim=1)

class MyDFFM3(nn.HybridBlock):
    def __init__(self, dims, inC, outC, unit):
        super(MyDFFM3, self).__init__()

        with self.name_scope():
            self.n = n = len(dims)
            self.be = MyBA('relu')
            self.ba = MyBA('relu')
            self.ip = MyIBA(unit, 'relu')
            self.ed = MyED(dims, inC, outC*n)

    def hybrid_forward(self, F, x):
        e = self.be(self.ed(x))

        deep = self.ip(e)
        order1 = F.sum(e, axis=2)

        ss, es = [], F.split(e, self.n, 1)
        for s in es: ss.append(F.split(s,self.n,2))

        order2 = []
        for i in xrange(self.n):
            temp = []
            for k in xrange(self.n):
                temp.append(ss[i][k]*ss[k][i])
            temp = F.concat(*temp, dim=1)
            order2.append(F.sum(temp, axis=1))
        order2 = F.concat(*order2, dim=1)

        return self.ba(
            F.concat(deep, order1, order2, dim=1))

###################################################

class MyModel(object):
    def __init__(self, gpu=0):
        self.gpu = gpu

    def getNet(self, ctx):
        raise NotImplementedError

    def getModel(self):
        return './model'

    def getEpoch(self):
        return 2

    def getMetric(self):
        raise NotImplementedError

    def getTrainer(self, params, iters):
        opt = mx.optimizer.SGD(
            wd=1e-3,
            momentum=0.9,
            learning_rate=1e-2,
            lr_scheduler=mx.lr_scheduler.FactorScheduler(
                iters*30, 1e-1, 1e-3
            )
        )

        return mx.gluon.Trainer(params, opt)

    def forDebug(self, out):
        pass

    def train(self):
        model = self.getModel()
        if not os.path.exists(model):
            os.mkdir(model)

        ctx = mx.gpu(self.gpu)
        net, myLoss = self.getNet(ctx)
        trainI, testI = self.getData(ctx)
        metric, monitor = self.getMetric()
        trainer = self.getTrainer(
            net.collect_params(), trainI.iters)

        logging.info('')
        result, epochs = 0, self.getEpoch()
        for epoch in xrange(1, epochs+1):
            logging.info('Epoch[%04d] start ...' % epoch)

            map(lambda x: x.reset(), [trainI, metric])
            for data, label in trainI:
                with autograd.record():
                    out = net.forward(data)
                    self.forDebug(out)
                    loss = myLoss(out, label)
                loss.backward()
                trainer.step(data.shape[0])

                metric.update(label, out)
            for name, value in metric.get():
                logging.info('Epoch[%04d] Train-%s=%f ...', epoch, name, value)

            _result = None
            map(lambda x: x.reset(), [testI, metric])
            for data, label in testI:
                out = net.forward(data)
                self.forDebug(out)
                metric.update(label, out)
            for name, value in metric.get():
                if name == monitor: _result = value
                logging.info('Epoch[%04d] Validation-%s=%f', epoch, name, value)

            if _result > result:
                result = _result
                name = '%s/%04d-%3.3f%%.params' % (model, epoch, result*100)
                net.save_params(name)
                logging.info('Save params to %s ...', name)

            logging.info('Epoch[%04d] done ...\n' % epoch)

class MyPredict(object):
    def __init__(self, gpu=0):
        self.gpu = gpu

    def getNet(self, ctx):
        raise NotImplementedError

    def preProcess(self, data):
        return data

    def postProcess(self, data, pData, output):
        raise NotImplementedError

    def onDone(self):
        pass

    def predict(self):
        ctx = mx.gpu(self.gpu)

        net = self.getNet(ctx)
        testI = self.getData(ctx)

        logging.info('Start predicting ...')

        testI.reset()
        for data in testI:
            pData = self.preProcess(data)
            out = net.forward(pData)
            self.postProcess(data, pData, out)

        self.onDone()
        logging.info('Prediction done ...')

###################################################

class MyLoss(nn.HybridBlock):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss = loss.SigmoidBCELoss()

    def hybrid_forward(self, F, pred, label):
        sampleWeight = 20*label+(1-label)
        return self.loss(pred, label, sampleWeight)

class MyMetric(object):
    def get(self):
        assert len(self.labels) != 0, 'Failed ...'
        assert len(self.outputs) != 0, 'Failed ...'

        labels = np.hstack(self.labels)
        outputs = np.hstack(self.outputs)
        results = (
            (
                'aucMetric',
                self.myAuc(labels, outputs)
            ),
            (
                'accMetric',
                self.myAcc(labels, outputs)
            ),
        )

        return results

    def myAuc(self, labels, outputs):
        return roc_auc_score(labels, outputs)

    def myAcc(self, labels, outputs):
        return np.mean(labels == (outputs>=0.5))

    def reset(self):
        self.labels = []
        self.outputs = []

    def update(self, label, output):
        label = label.asnumpy()
        output = mx.nd.sigmoid(output)
        output = output.asnumpy()[:,0]

        self.labels.append(label)
        self.outputs.append(output)

###################################################

class MyLoss2(nn.HybridBlock):
    def __init__(self):
        super(MyLoss2, self).__init__()
        self.loss = loss.SoftmaxCELoss()

    def hybrid_forward(self, F, pred, label):
        sampleWeight = 20*label+(1-label)
        return self.loss(pred, label, sampleWeight)

class MyMetric2(object):
    def get(self):
        assert len(self.labels) != 0, 'Failed ...'
        assert len(self.outputs) != 0, 'Failed ...'

        labels = np.hstack(self.labels)
        outputs = np.hstack(self.outputs)
        results = (
            (
                'aucMetric',
                self.myAuc(labels, outputs)
            ),
            (
                'accMetric',
                self.myAcc(labels, outputs)
            ),
        )

        return results

    def myAuc(self, labels, outputs):
        return roc_auc_score(labels, outputs)

    def myAcc(self, labels, outputs):
        return np.mean(labels == (outputs>=0.5))

    def reset(self):
        self.labels = []
        self.outputs = []

    def update(self, label, output):
        label = label.asnumpy()
        output = mx.nd.softmax(output)
        output = output.asnumpy()[:,1]

        self.labels.append(label)
        self.outputs.append(output)

###################################################

class MyLoss3(nn.HybridBlock):
    def __init__(self):
        super(MyLoss3, self).__init__()
        self.loss1 = loss.SoftmaxCELoss()
        self.loss2 = loss.SigmoidBCELoss()

    def hybrid_forward(self, F, pred, label):
        sampleWeight = 20*label+(1-label)

        pred1 = F.slice_axis(pred, 1, 0, 2)
        pred2 = F.slice_axis(pred, 1, 2, 3)

        loss1 = self.loss1(pred1, label, sampleWeight)
        loss2 = self.loss2(pred2, label, sampleWeight)

        return 0.5 * (loss1 + loss2)

class MyMetric3(object):
    def get(self):
        assert len(self.labels) != 0, 'Failed ...'
        assert len(self.outputs) != 0, 'Failed ...'

        labels = np.hstack(self.labels)
        outputs = np.hstack(self.outputs)
        results = (
            (
                'aucMetric',
                self.myAuc(labels, outputs)
            ),
            (
                'accMetric',
                self.myAcc(labels, outputs)
            ),
        )

        return results

    def myAuc(self, labels, outputs):
        return roc_auc_score(labels, outputs)

    def myAcc(self, labels, outputs):
        return np.mean(labels == (outputs>=0.5))

    def reset(self):
        self.labels = []
        self.outputs = []

    def update(self, label, output):
        label = label.asnumpy()

        output1 = output[:, 0:2]
        output2 = output[:, 2:3]

        output1 = mx.nd.softmax(output1)
        output1 = output1.asnumpy()[:,1]

        output2 = mx.nd.sigmoid(output2)
        output2 = output2.asnumpy()[:,0]

        output = 0.5*(output1 + output2)

        self.labels.append(label)
        self.outputs.append(output)

###################################################

