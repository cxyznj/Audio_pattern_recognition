import audioProcess
from random import shuffle
import torch
import numpy as np


def get_traindata(pos_path, neg_path, batch_size = 128):
    pos_files = audioProcess.getfilename(pos_path)
    neg_files = audioProcess.getfilename(neg_path)

    files = pos_files + neg_files
    shuffle(files)

    X = []
    y = []

    for i, file in enumerate(files):
        print("----Processing the %d file:%s----" % (i + 1, file))
        if file in pos_files:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=43)
        else:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=0, feature_len=43)

        # 音频长度短于阈值，舍弃之
        if len(label) <= 0:
            print("Drop file %s cause length of the voice is too short!" % file)
            continue

        X.extend(data)
        y.extend(label)

    print("Loading finish, divide into batches")

    index = list(range(len(y)))
    shuffle(index)

    Data = []
    Label = []

    k = int(len(y)/batch_size)

    for i in range(k):
        # 获取index中batch_size个索引，生成一批训练数据
        curdata = []
        curlabel = []
        for j in range(batch_size*i, batch_size*(i+1)):
            curdata.append([X[index[j]]])
            curlabel.append(y[index[j]])
        Data.append(curdata)
        Label.append(curlabel)

    print("Successful generate %d batch data for train!" %k)

    return Data, Label


def get_testdata(pos_path, neg_path, batch_size=128):
    pos_files = audioProcess.getfilename(pos_path)
    neg_files = audioProcess.getfilename(neg_path)

    files = pos_files + neg_files

    X = []
    y = []

    for i, file in enumerate(files):
        print("----Processing the %d file:%s----" % (i + 1, file))
        if file in pos_files:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=43)
        else:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=0, feature_len=43)

        # 音频长度短于阈值，舍弃之
        if len(label) <= 0:
            print("Drop file %s cause length of the voice is too short!" % file)
            continue

        X.extend(data)
        y.extend(label)

    print("Loading finish, divide into batches")

    Data = []
    Label = []

    k = int(len(y) / batch_size)

    for i in range(k):
        # 获取index中batch_size个索引，生成一批训练数据
        curdata = []
        curlabel = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            curdata.append([X[j]])
            curlabel.append(y[j])
        Data.append(curdata)
        Label.append(curlabel)

    print("Successful generate %d batch data for test!" % k)

    return Data, Label


def get_predictdata(audio_path):
    files = audioProcess.getfilename(audio_path)

    Data = []
    Label = []

    for i, file in enumerate(files):
        mfccs = audioProcess.get_feature(file)
        data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=43)

        # 音频长度短于阈值，舍弃之
        if len(label) <= 0:
            print("Drop file %s cause length of the voice is too short!" % file)
            continue

        # 规范化特征
        for i in range(label.shape[0]):
            data[i] = [data[i]]
        data = np.array(data)

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        Data.append(data)
        Label.append(label)
    print("calculate mfcc finished")

    return Data, Label, files
