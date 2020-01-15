# -*- coding: utf-8 -*-
"""
Created at 2019/12/17
@author: henk guo
"""

import os
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import os
digits = [str(i) for i in range(10)]
number = ['0', '1', '2', '3', '4',
          '5', '6', '7', '8', '9']


def extract_mfcc(input_path, file, num, nMel):
    y, sr = librosa.load(input_path + file)
    plt.figure(figsize=(3, 3), dpi=100)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('tw_data\\img\\' + num + '\\' + file[:-3] + 'png', bbox_inches='tight', pad_inches=-0.1)

    plt.close()
    return


for num in digits:
    print(num)
    count = 0  # number of files processed

    input_path = 'tw_data\\wav\\' + num + '\\'# input directory
    output_path = 'tw_data\\img\\' + num + '\\'

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for wavfile in os.listdir(input_path):

        # Input file
        S = extract_mfcc(input_path, wavfile, num, 256)

        # Count processed files
        count += 1
        if count % 50 == 0:
            print('%d files processed.' % count)

    print('Done!\t%d files processed.' % count)