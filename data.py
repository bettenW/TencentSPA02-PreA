#!/usr/bin/env python

import os
import csv
import json
import random
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from scipy.sparse import csr_matrix
from collections import defaultdict
from collections import OrderedDict

############################################

def mkdir(folder):
    folder = './datas/features/%s' % folder

    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def getMerged(*keys, **kwargs):
    kind = kwargs.get('kind', 0)
    name1 = './datas/trainMerged.csv'
    name2 = './datas/test1Merged.csv'

    for name in [name1, name2]:
        if os.path.exists(name): continue

        ad = pd.read_csv('./datas/adFeature.csv')
        df = pd.read_csv(name.replace('Merged', ''))
        user = pd.read_csv('./datas/userFeature.csv')

        notUsed = ['appIdAction', 'appIdInstall']
        user.drop(notUsed, axis=1, inplace=True)

        df = df.merge(ad).merge(user)
        if 'label' not in df: df['label'] = 0

        columns = sorted(df.columns.tolist())
        columns.remove('label'); columns.append('label')
        df.to_csv(name, index=False, columns=columns)

    name3 = './datas/trainMerged1.csv'
    name4 = './datas/trainMerged2.csv'

    exists = os.path.exists
    if not (exists(name3) and exists(name4)):
        random.seed(4567)
        t1n, t2n = int(7e+6), int(1e+6)

        df = pd.read_csv(name1)
        indices = range(len(df))
        random.shuffle(indices)
        indices3 = indices[:t1n]
        indices4 = indices[-t2n:]

        columns = sorted(df.columns.tolist())
        columns.remove('label'); columns.append('label')
        df.take(indices3).to_csv(
            name3, index=False, columns=columns)
        df.take(indices4).to_csv(
            name4, index=False, columns=columns)

    if not keys: keys = None
    if kind==1: return pd.read_csv(name1, usecols=keys)
    if kind==2: return pd.read_csv(name2, usecols=keys)
    if kind==0: return pd.concat([
        pd.read_csv(name1, usecols=keys),
        pd.read_csv(name2, usecols=keys),
    ], axis=0)
    if kind==3: return pd.read_csv(name3, usecols=keys)
    if kind==4: return pd.read_csv(name4, usecols=keys)

def getMergedA(*keys, **kwargs):
    kind = kwargs.get('kind', 0)
    name1 = './datas/trainMergedA.csv'
    name2 = './datas/test1MergedA.csv'

    for name in [name1, name2]:
        if os.path.exists(name): continue

        ad = pd.read_csv('./datas/adFeature.csv')
        df = pd.read_csv(name.replace('Merged', ''))
        user = pd.read_csv('./datas/userFeature.csv')

        df = df.merge(ad).merge(user)
        if 'label' not in df: df['label'] = 0

        columns = sorted(df.columns.tolist())
        columns.remove('label'); columns.append('label')
        df.to_csv(name, index=False, columns=columns)

    if not keys: keys = None
    if kind==1: return pd.read_csv(name1, usecols=keys)
    if kind==2: return pd.read_csv(name2, usecols=keys)
    if kind==0: return pd.concat([
        pd.read_csv(name1, usecols=keys),
        pd.read_csv(name2, usecols=keys),
    ], axis=0)

############################################

def getUF():
    name = './datas/userFeature.data'

    for f in open(name):
        user = {}
        for uf in f.strip().split('|'):
            uf = uf.split(' ')
            user[uf[0]] = map(int, uf[1:])

        yield user

def userMaxLen():
    folder = mkdir('msg')
    maxLen = defaultdict(int)
    maxFile = '%s/maxLen.txt' % folder

    if not os.path.exists(maxFile):
        for i, user in enumerate(getUF()):
            for key, vs in user.items():
                maxLen[key] = max(maxLen[key], len(vs))

        with open(maxFile, 'w') as f:
            for k in sorted(maxLen.keys()):
                f.write('%s %s\n' % (k, maxLen[k]))
            print 'Saved to %s ...' % saveName

    for line in open(maxFile):
        k, v = line.strip().split()
        maxLen[k] = int(v)

    return maxLen

def userFeature():
    default = { k: 0 for k in userMaxLen() }
    keys = sorted(default.keys())
    keys.remove('uid'); keys = ['uid'] + keys

    saveName = './datas/userFeature.csv'
    if os.path.exists(saveName): return

    with open(saveName, 'wb') as sf:
        dw = csv.DictWriter(sf, fieldnames=keys)
        dw.writeheader()

        for i, user in enumerate(getUF()):
            data = default.copy()
            for key, vs in user.items():
                data[key] = ' '.join(map(str,vs))
            dw.writerow(data)

            if (i+1) % 100000 == 0:
                print "%8d..." % (i+1)
        print 'Saved to %s ...' % saveName

############################################

def getFFM():
    folder = mkdir('ffm')
    datas = [
        ['trainMerged', 'trainFFM'],
        ['test1Merged', 'testFFM'],
    ]

    result, flag = [], True
    for _, save in datas:
        save = '%s/%s.txt' % (folder, save)
        result.append(save)

        if not os.path.exists(save):
            flag = False
    if flag: return result

    count, _  = setCount()
    fields = sorted(count.keys())

    result = []
    for name, save in datas:
        save = '%s/%s.txt' % (folder, save)
        result.append(save)
        if os.path.exists(save): continue

        txt = open(save, 'w')
        name = './datas/%s.csv' % name
        for i, line in enumerate(open(name)):
            vvs = line.strip().split(',')
            if i == 0: fNs = vvs; continue

            s = ''
            for k, vs in enumerate(vvs):
                if fNs[k] == 'label':
                    s = ('%d'%(vs=='1')) + s
                    continue

                ii = fields.index(fNs[k])
                for v in vs.split(' '):
                    c = count[fNs[k]][v]
                    if int(v) and c:
                        s += ' %d:%d:1' % (ii, c)
            txt.write('%s\n' % s)
        txt.close()

        print 'Saved to %s ...' % save
    return result

def splitFFM():
    t1, save3 = getFFM()

    folder = mkdir('ffm')
    exists = os.path.exists
    save1 = '%s/t1.txt' % folder
    save2 = '%s/t2.txt' % folder
    if exists(save1) and exists(save2):
        return save1, save2, save3

    random.seed(4567)
    lines = open(t1).readlines()
    random.shuffle(lines)
    t1n, t2n = int(7e+6), int(1e+6)

    with open(save1, 'w') as txt:
        txt.writelines(lines[:t1n])
    print 'Saved to %s ...' % save1
    with open(save2, 'w') as txt:
        txt.writelines(lines[-t2n:])
    print 'Saved to %s ...' % save2

    return save1, save2, save3

def getCount():
    folder = mkdir('msg')
    saveName = '%s/counts.json' % folder
    if os.path.exists(saveName):
        return json.load(open(saveName))

    datas = OrderedDict()
    df = getMerged(kind=0).drop('label', 1)
    for fn in df:
        if df[fn].dtype == 'object':
            d, ds = [], df[
                fn].str.split(' ', expand=False)
            for s in ds: d.extend(map(int, s))
            datas[fn] = Counter(d)
        else:
            datas[fn] = Counter(df[fn])
    json.dump(datas, open(saveName, 'w'))
    print 'Saved to %s ...' % saveName

    return datas

def setCount():
    count = getCount()
    fields = sorted(count.keys())

    offset = 0
    for k in fields:
        t = 5 if k=='uid' else 100

        temp = OrderedDict(
            sorted(
                count[k].items(),
                reverse=1, key=lambda i: i[1],
            )
        )
        count[k] = defaultdict(int, {
            str(u): i+1+offset
            for i, u in enumerate(temp)
            if temp[u]>=t and int(u)!=0
        })
        offset += len(count[k])

    return count, offset

def getFFMDim():
    t1, t2, t3 = splitFFM()

    folder = mkdir('msg')
    saveName = '%s/ffmDim.csv' % folder
    if os.path.exists(saveName):
        df = pd.read_csv(saveName)
        return OrderedDict(
            zip(df.columns, df.values[0]))

    datas = defaultdict(int)
    for t in [t1, t2, t3]:
        for line in open(t):
            temp = defaultdict(int)

            s = line.strip().split(' ')
            fs = map(int, map(
                lambda x: x.split(':')[0], s[1:]))

            for f in fs: temp['field#%03d'%f] += 1
            for key in temp:
                datas[key] = max(temp[key], datas[key])
    for key in datas: datas[key] = [datas[key]]

    pd.DataFrame.from_dict(datas).to_csv(
        saveName, index=False, columns=sorted(datas.keys()))
    print 'Saved to %s ...' % saveName

    return OrderedDict([
        (k, datas[k][0]) for k in sorted(datas.keys())])

def getSparse():
    offsets = [0]
    ffmDim = getFFMDim()
    for k in sorted(ffmDim.keys()):
        offsets.append(offsets[-1]+ffmDim[k])

    t1, t2, t3 = splitFFM()

    folder = mkdir('ffm')
    save1 = '%s/t1X.npz' % folder
    save2 = '%s/t2X.npz' % folder
    save3 = '%s/testX.npz' % folder

    ydfs = [
        [t1, save1], [t2, save2], [t3, save3],
    ]

    flag = True
    for t, save in ydfs:
        if not os.path.exists(save):
            flag = False
    if flag: return save1, save2, save3

    data, indices, indptr = [], [], [0]
    for k, (t, save) in enumerate(ydfs):
        ydfs[k].append(0)
        for line in open(t):
            ydfs[k][-1] += 1

            s = line.strip().split(' ')
            i = map(int, s[:1] + map(
                lambda x: x.split(':')[1], s[1:]))
            fs = map(int, map(
                lambda x: x.split(':')[0], s[1:]))

            indices.append(0)
            ios = [0] * len(offsets)
            for f in fs:
                ios[f] += 1
                indices.append(offsets[f] + ios[f])

            data.extend(i)
            indptr.append(indptr[-1] + len(i))

    rows, cols = len(indptr)-1, np.max(indices)+1
    csr = csr_matrix(
        (data, indices, indptr), shape=(rows, cols))

    start, end = 0, 0
    for t, save, l in ydfs:
        start, end = end, end + l
        sparse.save_npz(
            save, csr[start:end], compressed=True)

    return save1, save2, save3

def loadSparse(kind):
    return sparse.load_npz(getSparse()[kind])

############################################

if __name__ == '__main__':
    # print setCount()[1] # 90946
    getSparse()

