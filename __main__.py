import audioProcess
import torch
import Model
from random import shuffle
import os
import numpy as np

def train(pos_path, neg_path):
    model = Model.model(enum=3)
    if os.path.exists('model/cnn.pth'):
        model.load_model()

    Data, Label = get_traindata(pos_path, neg_path, batch_size=128)

    for i in range(len(Label)):
        data = Data[i]
        label = Label[i]

        data = np.array(data)
        label = np.array(label)

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)  # 转Float
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Long

        model.train_model(data, label)

        if ((i + 1) % 10 == 0):
            print("Save model in i=%d" % (i + 1))
            model.save_model()

    model.save_model()


def predict(audio_path):
    Data, Label, files = get_predictdata(audio_path)

    model = Model.model(enum=1)
    if os.path.exists('model/cnn.pth'):
        model.load_model()

    for i in range(len(files)):
        print("----Predict the %d file:%s----" % (i + 1, files[i]))
        result = model.predict(Data[i])
        print(result)


def test(audio_path):
    files = audioProcess.getfilename(audio_path)

    model = Model.model(enum=1)
    if os.path.exists('model/cnn.pth'):
        model.load_model()
    else:
        print("Have not model.")
        return

    for i, file in enumerate(files):
        print("----Testing the %d file:%s----" % (i + 1, file))
        mfccs = audioProcess.get_feature(audio_path + file)
        data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=32)

        # 音频长度短于阈值，舍弃之
        if len(label) <= 0:
            print("Drop file %s cause length of the voice is too short!" % file)
            continue

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)  # 转Double
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Double

        model.test_model(data, label)


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
            curdata.append([X[j]])
            curlabel.append(y[j])
        Data.append(curdata)
        Label.append(curlabel)

    print("Successful generate %d batch data!" %k)

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
        data = data.type(torch.FloatTensor)  # 转Double
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Double

        Data.append(data)
        Label.append(label)
    print("calculate mfcc finished")

    return Data, Label, files


if __name__ == "__main__":
    predict("audio/test")
    # test("audio/test/")
    # train("audio/human", "audio/noise")