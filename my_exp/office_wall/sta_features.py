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


acc_gyro = []
gob='good'
# read acc data oaver z
for i in range(1, 101): # number of examples
    acc_gyro_pattern = []
    for j in range(1, 6): # number of patterns
            pattern = str(j*10)
            fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
            folder = 'TestData_'+gob+'_wall_'+pattern
            ls = './'+folder + '/'+fileName+'.txt' 
            acc_z= []
            time = []
            with open(ls, 'r') as f:
                lines = f.readlines()
                num =0
                for line in lines[:-1]:
                    num = num + 1
                    if num>1:
                       value = line.rstrip().split(',')
                       acc_z.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
                       time.append(value[3])
            startTime = time[0].split(' ')[1].split('.')[0].split(':')
            startValue = int(startTime[0])*60*60+int(startTime[1])*60+int(startTime[-1])+int(time[0].split('.')[1])*0.001
            listTime_acc=[]
            for tm in time:
                va = tm.split(' ')[-1].split('.')[0].split(':')
                total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
                listTime_acc.append(total - startValue)
            acc_z = np.array(acc_z)
            acc_MAV = np.mean(abs(acc_z))
            acc_var = np.var(acc_z)
            acc_RMS = np.sqrt(np.mean(acc_z**2))
            acc_std = np.std(acc_z)
            acc_MAD = np.median(np.abs(acc_z-np.median(acc_z)))
            acc_skewness = stats.skew(acc_z)
            acc_kurtosis = stats.kurtosis(acc_z)
            acc_iqr = stats.iqr(acc_z)
            acc_energy = np.mean(acc_z**2)
            #acc_entropy = stats.entropy(acc_z, base=2)
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
                    if num>1:
                       value = line.rstrip().split(',')
                       gyro_z.append(float(value[2]))
                       time.append(value[3])
                startTime = time[0].split(' ')[1].split('.')[0].split(':')
                startValue = int(startTime[0])*60*60+int(startTime[1])*60+int(startTime[-1])+int(time[0].split('.')[1])*0.001
                listTime_gyro=[]
            for tm in time:
                va = tm.split(' ')[-1].split('.')[0].split(':')
                total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
                listTime_gyro.append(total - startValue)
            gyro_z = np.array(gyro_z)
            gyro_MAV = np.mean(abs(gyro_z))
            gyro_var = np.var(gyro_z)
            gyro_RMS = np.sqrt(np.mean(gyro_z**2))
            gyro_std = np.std(gyro_z)
            gyro_MAD = np.median(np.abs(gyro_z-np.median(gyro_z)))
            gyro_skewness = stats.skew(gyro_z)
            gyro_kurtosis = stats.kurtosis(gyro_z)
            gyro_iqr = stats.iqr(gyro_z)
            gyro_energy = np.mean(gyro_z**2)
            #gyro_entropy = stats.entropy(gyro_z, base=2)
            acc_gyro_pattern.append([acc_MAV, acc_var, acc_RMS, acc_std, acc_MAD, acc_skewness, acc_kurtosis, acc_iqr, acc_energy, gyro_MAV, gyro_var, gyro_RMS, gyro_std, gyro_MAD, gyro_skewness, gyro_kurtosis, gyro_iqr, gyro_energy])
    acc_gyro.append(acc_gyro_pattern)
len_ag = len(acc_gyro)
acc_gyro = np.array(acc_gyro)
acc_gyro = np.reshape(acc_gyro, [len_ag, 9*2*5])
dbfile = open(gob+'_wall.pkl', 'wb')
pickle.dump(acc_gyro, dbfile)
dbfile.close()
