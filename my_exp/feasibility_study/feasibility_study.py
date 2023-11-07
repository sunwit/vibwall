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


gob='good'
# read acc data oaver z
for i in range(1, 2): # number of examples
    acc_pattern = []
    gyro_pattern = []
    for j in range(1, 6): # number of patterns
            pattern = str(j*10)
            fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
            folder = 'TestData_'+gob
            ls = './'+folder + '/'+fileName+'.txt' 
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1:
                       value = line.rstrip().split(',')
                       acc_pattern.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
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
gob='bad'
# read acc data oaver z
for i in range(1, 2): # number of examples
    acc_pattern_1 = []
    gyro_pattern_1 = []
    for j in range(1, 6): # number of patterns
            pattern = str(j*10)
            fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
            folder = 'TestData_'+gob
            ls = './'+folder + '/'+fileName+'.txt' 
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1:
                       value = line.rstrip().split(',')
                       acc_pattern_1.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
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
                       gyro_pattern_1.append(float(value[2]))
 
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(acc_pattern, label='Wall w/o cracks')
    plt.plot(acc_pattern_1, label='Wall w/ cracks')
    plt.legend(fontsize=15)
    plt.xlabel('Sample index', fontsize=18)
    plt.ylabel('Acc.z($m/s^2$)', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.xlim([-25, 25])
    plt.ylim([-1, 1])
    #plt.savefig('fs_acc.pdf')
    fig.subplots_adjust(left=0.25)
    fig.subplots_adjust(bottom=0.15)
    plt.show()

    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(gyro_pattern, label='Wall w/o cracks')
    plt.plot(gyro_pattern_1, label='Wall w/ cracks')
    plt.legend(fontsize=15)
    plt.xlabel('Sample index', fontsize=18)
    plt.ylabel('Gyro.z($rad/s$)', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.xlim([-25, 25])
    plt.ylim([-5, 0])
    plt.savefig('fs_gyro.pdf')
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(bottom=0.15)
    plt.show()

