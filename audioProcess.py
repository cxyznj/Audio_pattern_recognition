import librosa
import os
import numpy as np
import pydub

def get_feature(filename):
    y, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = mfccs.T
    return mfccs


def process_mfccs(mfccs, type = 0, feature_len = 43):
    n, m = mfccs.shape
    # 43行特征值代表一秒，以一秒为基础单位
    # 32个特征 -> 0.743s
    k = int(n/feature_len)
    data = []
    for i in range(k):
        # mfcc = [mfccs[(i*feature_len):((i+1)*feature_len)]]
        mfcc = mfccs[(i*feature_len):((i+1)*feature_len)]
        data.append(mfcc)

    # data = np.array(data)

    if type == 0:
        label = np.zeros(k)
    else:
        label = np.ones(k)

    return data, label


def getfilename(file_dir):
    #print("----begin----")
    filelist = []
    for root, dirs, files in os.walk(file_dir):
        #print("root = ", root)
        #print("dirs = ", dirs)
        #print("files = ", files)
        for file in files:
            filelist.append(root + '/' + file)

        if len(dirs) > 0:
            for dir in dirs:
                dirfiles = getfilename(root + '/' + dir)
                filelist.extend(dirfiles)

        break

    #print("filelist = ", filelist)

    return filelist


def cut_audio(wav_path, part_wav_path, start_time, end_time):
    addition_time = 240
    start_time = int(start_time)
    end_time = int(end_time)
    if start_time > addition_time:
        start_time -= addition_time
    else:
        start_time = 0
    end_time += addition_time

    sound = pydub.AudioSegment.from_mp3(wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")


if __name__ == "__main__":
    pass
    #mfccs = get_feature("audio/test/LandingGuy2.wav")
    #process_mfccs(mfccs)
    #file_name("audio/test/")