from datetime import datetime
from sqlite3 import Timestamp
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
#from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks
import scipy.integrate
from scipy import stats
import pickle
from scipy import spatial
from python_speech_features import mfcc

# read acc/gyro data oaver z
base = 0
simi = []
for i in range(1, 6): # number of examples
    base = base + 1
    acc_gyro_pattern = []
    for j in range(1, 6): # number of patterns
            gyro_pattern = []
            acc_pattern = []
            pattern = str(j*10)
            fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
            folder = 'TestData'
            ls = './'+folder + '/'+fileName+'.txt' 
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1 and num<=500:
                       value = line.rstrip().split(',')
                       acc_pattern.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
                while(num<500):
                     acc_pattern.append(acc_patten[-1])
                     num +=1
            acc_pattern = np.array(acc_pattern)
            # mfcc features
            acc_mfcc_feat = mfcc(acc_pattern)
            # read gyro data oaver z
            fileName = 'GyroscopeFrequency'+pattern+'_'+str(i)
            ls = './'+folder + '/'+fileName+'.txt' 
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1 and num<=500:
                       value = line.rstrip().split(',')
                       gyro_pattern.append(float(value[2]))
                while(num<500):
                    gyro_pattern.append(gyro_pattern[-1])
                    num+=1
            gyro_pattern = np.array(gyro_pattern)
            # mfcc feature
            gyro_mfcc_feat = mfcc(gyro_pattern)
            acc_gyro_pattern = np.concatenate([acc_mfcc_feat.flatten(), gyro_mfcc_feat.flatten()], axis=0)
            # faltten mfcc feature
    if base==1:
       base_pattern = acc_gyro_pattern
    else:
       cosine_dis = spatial.distance.cosine(base_pattern, acc_gyro_pattern)
       simi.append(1-cosine_dis)
print(simi)
#
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(['Postion 1', 'Position 2', 'Position 3'], simi[:-1], color='b', width=0.5)
plt.axhline(y=1, color='r', linestyle=':')
#plt.legend(fontsize=15)
plt.xlabel('Smartphone position', fontsize=18)
plt.ylabel('Similarity', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim([-25, 25])
#plt.ylim([-1, 1])
#plt.savefig('position_similarity.pdf')
#fig.subplots_adjust(left=0.25)
#fig.subplots_adjust(bottom=0.15)
plt.show()
