import librosa
import os
import numpy as np
import pydub
from aip import AipSpeech

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
    addition_time = 495
    start_time = int(start_time)
    end_time = int(end_time)
    if start_time > addition_time:
        start_time -= addition_time
    else:
        start_time = 0
    end_time += addition_time

    sound = pydub.AudioSegment.from_wav(wav_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav", codec="pcm", bitrate='256k')
    # word.export(part_wav_path, format="wav", bitrate='32k')

def voice_to_text(filename):
    APP_ID = '16242951'
    API_KEY = 'Lblvx1OnWkjvNdNiIczGnoGP'
    SECRET_KEY = 'ErY9rSO3Sz0VGMGzo9UbBFzZDDfSwbW3'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    # 读取文件
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    # 识别本地文件
    result = client.asr(get_file_content(filename), 'wav', 16000, {
        'dev_pid': 1537,
    })

    filename = filename[:-4] + '.txt'

    with open(filename, 'w') as fileobject:
        if result.get('err_msg') == 'success.':
            fileobject.write(result.get('result')[0])
        else:
            fileobject.write("err_msg:" + result.get('err_msg') + '\n' + "err_no:" + str(result.get("err_no")))
        fileobject.close()

    return result

if __name__ == "__main__":
    pass
    #mfccs = get_feature("audio/test/LandingGuy2.wav")
    #process_mfccs(mfccs)
    #file_name("audio/test/")