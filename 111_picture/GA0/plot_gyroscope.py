from datetime import datetime
from sqlite3 import Timestamp
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.fftpack import fft, fftfreq
from scipy import signal
from scipy.signal import savgol_filter

# read gyroscope data over x, y, z

# in the air
gyro_z_0 = []
acc_z_0 = []
pattern = '10'
with open('../GA0/GyroscopeFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         gyro_z_0.append(float(value[2]))
with open('../GA0/GAcceleratorFrequency' + pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         acc_z_0.append(float(value[2]))
# on the good wall
gyro_z_1 = []
acc_z_1 = []
with open('../GA1/GyroscopeFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         gyro_z_1.append(float(value[2]))
with open('../GA1/GAcceleratorFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         acc_z_1.append(float(value[2]))

# on the bad wall
gyro_z_2 = []
acc_z_2 = []
with open('../GA2/GyroscopeFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         gyro_z_2.append(float(value[2]))
with open('../GA2/GAcceleratorFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         acc_z_2.append(float(value[2]))
# detrend the signals
N_gyro_z_0 = len(gyro_z_0)
T_gyro_z_0 = 1.0/N_gyro_z_0
N_gyro_z_1 = len(gyro_z_1)
T_gyro_z_1 = 1.0/N_gyro_z_1
N_gyro_z_2 = len(gyro_z_2)
T_gyro_z_2 = 1.0/N_gyro_z_2


gyro_z_0 = signal.detrend(gyro_z_0)
gyro_z_1 = signal.detrend(gyro_z_1)
gyro_z_2 = signal.detrend(gyro_z_2)

acc_z_0 = signal.detrend(acc_z_0)
acc_z_1 = signal.detrend(acc_z_1)
acc_z_2 = signal.detrend(acc_z_2)


gyro_z_0 = fft(gyro_z_0)
gyro_z_0_f = fftfreq(N_gyro_z_0, T_gyro_z_0)[:N_gyro_z_0//2]

gyro_z_1 = fft(gyro_z_1)
gyro_z_1_f = fftfreq(N_gyro_z_1, T_gyro_z_1)[:N_gyro_z_1//2]

gyro_z_2 = fft(gyro_z_2)
gyro_z_2_f = fftfreq(N_gyro_z_2, T_gyro_z_2)[:N_gyro_z_2//2]

# gyroscope-z data plot
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(gyro_z_0_f, 2.0/N_gyro_z_0*np.abs(gyro_z_0[0:N_gyro_z_0//2]), label='Wall without cracks')
plt.plot(gyro_z_1_f, 2.0/N_gyro_z_1*np.abs(gyro_z_1[0:N_gyro_z_1//2]), label='Wall with cracks')
#plt.plot(gyro_z_2_f, 2.0/N_gyro_z_2*np.abs(gyro_z_2[0:N_gyro_z_2//2]), label='Attaching to the wall with cracks')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('FFT bins', fontsize=18)
plt.ylabel('FFT value', fontsize=18)
plt.ylim([0, 0.025])
fig.subplots_adjust(left=0.15)
plt.savefig('gyro_fft.pdf')
plt.show()


#acc-z 
N_acc_z_0 = len(acc_z_0)
T_acc_z_0 = 1.0/N_acc_z_0
acc_z_0 = fft(acc_z_0)
acc_z_0_f = fftfreq(N_acc_z_0, T_acc_z_0)[:N_acc_z_0//2]

N_acc_z_1 = len(acc_z_1)
T_acc_z_1 = 1.0/N_acc_z_1
acc_z_1 = fft(acc_z_1)
acc_z_1_f = fftfreq(N_acc_z_1, T_acc_z_1)[:N_acc_z_1//2]

N_acc_z_2 = len(acc_z_2)
print(N_acc_z_2)
T_acc_z_2 = 1.0/N_acc_z_2
acc_z_2 = fft(acc_z_2)
acc_z_2_f = fftfreq(N_acc_z_2, T_acc_z_2)[:N_acc_z_2//2]

# accelerometer-z data plot
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(acc_z_0_f, 2.0/N_acc_z_0*np.abs(acc_z_0[0:N_acc_z_0//2]), label='Wall without cracks')
plt.plot(acc_z_1_f, 2.0/N_acc_z_1*np.abs(acc_z_1[0:N_acc_z_1//2]), label='Wall with cracks')
#plt.plot(acc_z_2_f, 2.0/N_acc_z_2*np.abs(acc_z_2[0:N_acc_z_2//2]), label='Attaching to the wall with cracks')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('FFT bins', fontsize=18)
plt.ylabel('FFT value', fontsize=18)
#plt.xlim([600, 1300])
fig.subplots_adjust(left=0.15)
plt.savefig('acc_fft.pdf')
plt.show()
