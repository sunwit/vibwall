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
                    if num>1:
                       value = line.rstrip().split(',')
                       acc_pattern.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
            acc_pattern = np.array(acc_pattern)
            # statistical features
            acc_MAV = np.mean(abs(acc_pattern))
            acc_var = np.var(acc_pattern)
            acc_RMS = np.sqrt(np.mean(acc_pattern**2))
            acc_std = np.std(acc_pattern)
            acc_MAD = np.median(np.abs(acc_pattern-np.median(acc_pattern)))
            acc_skewness = stats.skew(acc_pattern)
            acc_kurtosis = stats.kurtosis(acc_pattern)
            acc_iqr = stats.iqr(acc_pattern)
            acc_energy = np.mean(acc_pattern**2)                       
            # read gyro data oaver z
            fileName = 'GyroscopeFrequency'+pattern+'_'+str(i)
            ls = './'+folder + '/'+fileName+'.txt' 
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1:
                       value = line.rstrip().split(',')
                       gyro_pattern.append(float(value[2]))
            gyro_pattern = np.array(gyro_pattern)
            gyro_MAV = np.mean(abs(gyro_pattern))
            gyro_var = np.var(gyro_pattern)
            gyro_RMS = np.sqrt(np.mean(gyro_pattern**2))
            gyro_std = np.std(gyro_pattern)
            gyro_MAD = np.median(np.abs(gyro_pattern-np.median(gyro_pattern)))
            gyro_skewness = stats.skew(gyro_pattern)
            gyro_kurtosis = stats.kurtosis(gyro_pattern)
            gyro_iqr = stats.iqr(gyro_pattern)
            gyro_energy = np.mean(gyro_pattern**2)
            acc_gyro_pattern.append([acc_MAV, acc_var, acc_RMS, acc_std, acc_MAD, acc_skewness, acc_kurtosis, acc_iqr, acc_energy, gyro_MAV, gyro_var, gyro_RMS, gyro_std, gyro_MAD, gyro_skewness, gyro_kurtosis, gyro_iqr, gyro_energy])
    if base==1:
       base_pattern = acc_gyro_pattern[0]
    else:
       cosine_dis = spatial.distance.cosine(base_pattern, acc_gyro_pattern[0])
       eucli_dis = spatial.distance.euclidean(base_pattern, acc_gyro_pattern[0])
       mahan_dis = spatial.distance.cityblock(base_pattern, acc_gyro_pattern[0])
       simi.append(1-cosine_dis)
simi_mfcc=[0.9114511364987412, 0.9180137581095408, 0.895813277912788, 0.879799489785815]

fig, ax = plt.subplots(figsize=(8,6))
X = np.array([0, 2, 4, 6])
ax.bar(X, simi, color='b', width=0.25, label='Statistical feature')
ax.bar(X+0.25, simi_mfcc, color='r', width=0.25, label='MFCC feature')
plt.xticks([r+0.125 for r in [0, 2, 4, 6]], ['Orientation 1', 'Orientation 2', 'Orientation 3', 'Orientation 4'])
#plt.axhline(y=1, color='r', linestyle=':')
plt.legend(loc=3, fontsize=18)
plt.xlabel('Smartphone$\'$s orientation', fontsize=18)
plt.ylabel('Similarity', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim([-25, 25])
#plt.ylim([-1, 1])
plt.savefig('position_similarity.pdf')
#fig.subplots_adjust(left=0.25)
#fig.subplots_adjust(bottom=0.15)
plt.show()
