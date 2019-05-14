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
    else:
        print("Warning: Do not have model.")

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
    else:
        print("Do not have model.")
        return

    for i in range(len(files)):
        print("----Predict the %d file:%s----" % (i + 1, files[i]))
        result = model.predict(Data[i])
        result = result[0].numpy()
        print(result)

        start_time = 0
        end_time = 0
        start_flag = False
        for j in range(len(result)):
            if(result[j] == 1):
                if not start_flag:
                    start_flag = True
                    start_time = end_time = j
            else:
                if start_flag:
                    # 允许有一秒噪声容忍
                    if j < len(result)-1:
                        if result[j+1] == 1:
                            continue
                
                    start_flag = False
                    end_time = j
                    print("time = ", start_time, end_time)
                    fname = files[i][:-4] + 'output' + '/' + str(start_time) + '_' + str(end_time) + '.wav'
                    if not os.path.exists(files[i][:-4] + 'output'):
                        os.mkdir(files[i][:-4] + 'output')
                    audioProcess.cut_audio(files[i], fname, start_time*1000, end_time*1000)
                    audioProcess.voice_to_text(fname)
        if start_flag:
            end_time = len(result)
            print("time =", start_time, end_time)
            fname = files[i][:-4] + 'output' + '/' + str(start_time) + '_' + str(end_time) + '.wav'
            if not os.path.exists(files[i][:-4] + 'output'):
                os.mkdir(files[i][:-4] + 'output')
            audioProcess.cut_audio(files[i], fname, start_time * 1000, end_time * 1000)
            audioProcess.voice_to_text(fname)



def test(pos_path, neg_path):
    model = Model.model(enum=3)
    if os.path.exists('model/cnn.pth'):
        model.load_model()
    else:
        print("Do not have model.")
        return

    Data, Label = get_testdata(pos_path, neg_path, batch_size=128)

    for i in range(len(Label)):
        data = Data[i]
        label = Label[i]

        data = np.array(data)
        label = np.array(label)

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)  # 转Float
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Long

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
        data = data.type(torch.FloatTensor)  # 转Double
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Double

        Data.append(data)
        Label.append(label)
    print("calculate mfcc finished")

    return Data, Label, files


if __name__ == "__main__":
    # train("audio/human", "audio/noise")
    predict("audio/test")
    # test("audio/human", "audio/noise")