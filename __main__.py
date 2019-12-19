import audioProcess
import DataProcess
import torch
import Model
import os
import numpy as np
import time


def train(pos_path, neg_path, enum = 1):
    model = Model.model(enum=enum)
    if os.path.exists('model/cnn.pth'):
        model.load_model()
    else:
        print("Warning: Do not have model.")

    if os.path.exists('audio/Preprocessed/traindata.npy') and os.path.exists('audio/Preprocessed/trainlabel.npy'):
        print("Loading data set from file.")
        Data = np.load('audio/Preprocessed/traindata.npy')
        Label = np.load('audio/Preprocessed/trainlabel.npy')
    else:
        Data, Label = DataProcess.get_traindata(pos_path, neg_path, batch_size=128)

        Data = np.array(Data)
        Label = np.array(Label)

        np.save('audio/Preprocessed/traindata.npy', Data)
        np.save('audio/Preprocessed/trainlabel.npy', Label)

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
            print("Save model in i=%d." % (i + 1))
            model.save_model()

    model.save_model()


def predict(audio_path):
    Data, Label, files = DataProcess.get_predictdata(audio_path)

    model = Model.model(enum=1)
    if os.path.exists('D:/WorkSpace/Study/实验室/19_audio_recognition/code/model/cnn.pth'):
        model.load_model('D:/WorkSpace/Study/实验室/19_audio_recognition/code/model/cnn.pth')
    else:
        print("Error: Do not have model.")
        return

    rt_duration = []

    for i in range(len(files)):
        print("----Predict the %d file:%s----" % (i + 1, files[i]))
        bg_time = time.time()
        result = model.predict(Data[i])
        result = result[0].numpy()
        print(result)
        cur_duration = []

        start_time = 0
        start_flag = False
        for j in range(len(result)):
            if(result[j] == 1):
                if not start_flag:
                    start_flag = True
                    start_time = j
            else:
                if start_flag:
                    # 允许有一秒噪声容忍
                    if j < len(result)-1:
                        if result[j+1] == 1:
                            continue
                    
                    start_flag = False
                    end_time = j
                    # 去除小于一秒的数据
                    if end_time - start_time <= 1:
                        continue
                        
                    print("human voice in["+str(start_time)+','+str(end_time)+']')
                    cur_duration.append([start_time, end_time])
                    #fname = files[i][:-4] + 'output' + '/' + str(start_time) + '_' + str(end_time) + '.wav'
                    #if not os.path.exists(files[i][:-4] + 'output'):
                    #    os.mkdir(files[i][:-4] + 'output')
                    #audioProcess.cut_audio(files[i], fname, start_time*1000, end_time*1000)
                    #audioProcess.voice_to_text(fname)
        if start_flag:
            end_time = len(result)
            if end_time - start_time > 1:
                print("human voice in["+str(start_time)+','+str(end_time)+']')
                cur_duration.append([start_time, end_time])
                #fname = files[i][:-4] + 'output' + '/' + str(start_time) + '_' + str(end_time) + '.wav'
                #if not os.path.exists(files[i][:-4] + 'output'):
                #    os.mkdir(files[i][:-4] + 'output')
                #audioProcess.cut_audio(files[i], fname, start_time * 1000, end_time * 1000)
                #audioProcess.voice_to_text(fname)

        print("Used time = %.6fs" %(time.time()-bg_time))
        rt_duration.append(cur_duration)
    return rt_duration


def test(pos_path, neg_path):
    model = Model.model(enum=1)
    if os.path.exists('model/cnn_m15.pth'):
        model.load_model('model/cnn_m15.pth')
    else:
        print("Do not have model.")
        return

    if os.path.exists('audio/Preprocessed/testdata.npy') and os.path.exists('audio/Preprocessed/testlabel.npy'):
        print("Loading Data from file!")
        Data = np.load('audio/Preprocessed/testdata.npy')
        Label = np.load('audio/Preprocessed/testlabel.npy')
    else:
        Data, Label = DataProcess.get_testdata(pos_path, neg_path, batch_size=128)

        Data = np.array(Data)
        Label = np.array(Label)

        np.save('audio/Preprocessed/testdata.npy', Data)
        np.save('audio/Preprocessed/testlabel.npy', Label)

    loss = .0
    acc = .0

    for i in range(len(Label)):
        data = Data[i]
        label = Label[i]

        data = np.array(data)
        label = np.array(label)

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)  # 转Float
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Long

        curloss, curacc = model.test_model(data, label)
        loss += curloss
        acc += curacc
    loss /= len(Label)
    acc /= len(Label)
    print("Average test loss: %.6f, test accuracy: %.6f" %(loss, acc))
    
def hello(str):
    print("Hello" + str)
    return str


if __name__ == "__main__":
    # train("audio/trainhuman", "audio/trainnoise", 20)
    #print(predict("D:/PyCharm_Files/Audio_pattern_recognition/audio/beiguo.wav"))
    print(predict("mic3.wav"))
    # test("audio/testhuman", "audio/testnoise")