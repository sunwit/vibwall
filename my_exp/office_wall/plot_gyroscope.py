from datetime import datetime
from sqlite3 import Timestamp
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy import signal
from scipy.signal import savgol_filter
#from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks



# read acc data oaver z
folder = 'TestData_good_wall_30'
filepath = 'GAcceleratorFrequency30_'
for i in range(1, 10):
    ls = './'+folder + '/'+filepath+str(i)+'.txt' 
    acc_z= []
    with open(ls, 'r') as f:
        lines = f.readlines()
        num =0
        for line in lines[:-1]:
            num = num + 1
            if num>1:
               value = line.rstrip().split(',')
               acc_z.append(float(value[2]))
    """
    cur_len = len(acc_z)
    gap_len = 1024-cur_len
    while(gap_len>0):
      acc_z.append(0)
      gap_len-=1
    #acc_de = signal.detrend(acc_z)
    """
    acc_de = acc_z

    len_acc = len(acc_de)
    T_acc = 1/100.0
    acc = np.fft.fft(acc_de)
    acc_f = np.fft.fftfreq(len_acc, d=T_acc)
    # find peaks
    height_threshold = 10
    peaks_index, properties = find_peaks(acc[:len_acc//2], height=height_threshold)
    print('positions of frequency peaks:')
    for i in range(len(peaks_index)):
        print("%4.4f "  % acc_f[peaks_index[i]])
    plt.plot(acc_f, acc, '-', acc_f[peaks_index], properties['peak_heights'], 'x')
    plt.show()

"""    
# read gyro data oaver z
folder = 'TestData_good_wall_10'
filepath = 'GyroscopeFrequency10_'
for i in range(1, 101):
    ls = './'+folder + '/'+filepath+str(i)+'.txt' 
    gyro_z= []
    with open(ls, 'r') as f:
        lines = f.readlines()
        num =0
        for line in lines[:-1]:
            num = num + 1
            if num>1:
               value = line.rstrip().split(',')
               gyro_z.append(float(value[2]))
    cur_len = len(gyro_z)
    gap_len = 1024-cur_len
    while(gap_len>0):
      gyro_z.append(0)
      gap_len-=1
    gyro_de = signal.detrend(gyro_z)
    len_gyro = len(gyro_de)
    T_gyro = 1.0/len_gyro
    gyro = fft(gyro_de)
    gyro_f = fftfreq(len_gyro, T_gyro)[:len_gyro//2]

fig, ax = plt.subplots(figsize=(8,6))
plt.plot(acc_f, 2.0/len_acc*np.abs(acc[0:len_acc//2]))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('FFT bins', fontsize=18)
plt.ylabel('FFT value', fontsize=18)
fig.subplots_adjust(left=0.15)
#plt.savefig('gyro_fft.pdf')
plt.show()    


fig, ax = plt.subplots(figsize=(8,6))
plt.plot(gyro_f, 2.0/len_gyro*np.abs(gyro[0:len_gyro//2]))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('FFT bins', fontsize=18)
plt.ylabel('FFT value', fontsize=18)
fig.subplots_adjust(left=0.15)
#plt.savefig('gyro_fft.pdf')
plt.show()    
"""
