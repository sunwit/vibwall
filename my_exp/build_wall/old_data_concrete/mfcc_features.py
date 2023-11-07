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
from python_speech_features import mfcc

acc_gyro = [] #np.zeros([100, 2, 5, 124, 13])
gob='bad_wall'
# read acc data oaver z
for i in range(1, 101): # number of examples
    acc_gyro_pattern = []
    for j in range(1, 6): # number of patterns
            pattern = str(j*10)
            fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
            folder = 'TestData_concrete_with_cracks_1cm'
            ls = './'+folder + '/'+fileName+'.txt' 
            acc_z= []
            time = []
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1 and num<=500:
                       value = line.rstrip().split(',')
                       acc_z.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
                       time.append(value[3])
                while(num<500):
                    acc_z.append(acc_z[-1])
                    num = num + 1
            startTime = time[0].split(' ')[1].split('.')[0].split(':')
            startValue = int(startTime[0])*60*60+int(startTime[1])*60+int(startTime[-1])+int(time[0].split('.')[1])*0.001
            listTime_acc=[]
            for tm in time:
                va = tm.split(' ')[-1].split('.')[0].split(':')
                total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
                listTime_acc.append(total - startValue)
            acc_z = np.array(acc_z)
            acc_mfcc_feat=mfcc(acc_z, 100)
            # read gyro data oaver z
            fileName = 'GyroscopeFrequency'+pattern+'_'+str(i)
            ls = './'+folder + '/'+fileName+'.txt' 
            gyro_z= []
            time = []
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1 and num<=500:
                       value = line.rstrip().split(',')
                       gyro_z.append(float(value[2]))
                       time.append(value[3])
                while(num<500):    
                    gyro_z.append(gyro_z[-1])
                    num = num + 1
                startTime = time[0].split(' ')[1].split('.')[0].split(':')
                startValue = int(startTime[0])*60*60+int(startTime[1])*60+int(startTime[-1])+int(time[0].split('.')[1])*0.001
                listTime_gyro=[]
            for tm in time:
                va = tm.split(' ')[-1].split('.')[0].split(':')
                total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
                listTime_gyro.append(total - startValue)
            gyro_z = np.array(gyro_z)
            gyro_mfcc_feat = mfcc(gyro_z, 100)
            acc_gyro_pattern.append([acc_mfcc_feat, gyro_mfcc_feat])
    acc_gyro.append(acc_gyro_pattern)
len_ag = len(acc_gyro)
acc_gyro = np.array(acc_gyro)
acc_gyro = np.reshape(acc_gyro, [100, 2, 5, 497, 13])
dbfile = open(gob+'_wall_mfcc.pkl', 'wb')
pickle.dump(acc_gyro, dbfile)
dbfile.close()
