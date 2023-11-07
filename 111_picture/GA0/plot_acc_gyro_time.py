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
gyro_x_0 = []
gyro_y_0 = []
gyro_z_0 = []
acc_x_0 = []
acc_y_0 = []
acc_z_0 = []
time_gyro_0 = []
time_acc_0 = []
pattern = '10'
with open('../GA0/GyroscopeFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         gyro_x_0.append(float(value[0]))
         gyro_y_0.append(float(value[1]))
         gyro_z_0.append(float(value[2]))
         time_gyro_0.append(value[3])
     startTime = time_gyro_0[0].split(' ')[1].split('.')[0].split(':')
     startValue = int(startTime[0])*60*60 + int(startTime[1])*60 + int(startTime[-1]) + int(time_gyro_0[0].split('.')[1])*0.001
     list_time_gyro_0 = []
     for tm in time_gyro_0:
        va = tm.split(' ')[-1].split('.')[0].split(':')
        total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
        list_time_gyro_0.append(total - startValue)
with open('../GA0/GAcceleratorFrequency' + pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         acc_x_0.append(float(value[0]))
         acc_y_0.append(float(value[1]))
         acc_z_0.append(float(value[2]))
         time_acc_0.append(value[3])
     startTime = time_acc_0[0].split(' ')[1].split('.')[0].split(':')
     startValue = int(startTime[0])*60*60 + int(startTime[1])*60 + int(startTime[-1]) + int(time_acc_0[0].split('.')[1])*0.001
     list_time_acc_0 = []
     for tm in time_acc_0:
        va = tm.split(' ')[-1].split('.')[0].split(':')
        total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
        list_time_acc_0.append(total - startValue)

# on the good wall
gyro_x_1 = []
gyro_y_1 = []
gyro_z_1 = []

acc_x_1 = []
acc_y_1 = []
acc_z_1 = []

time_acc_1 = []
time_gyro_1 = []

with open('../GA1/GyroscopeFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         gyro_x_1.append(float(value[0]))
         gyro_y_1.append(float(value[1]))
         gyro_z_1.append(float(value[2]))
         time_gyro_1.append(value[3])
     startTime = time_gyro_1[0].split(' ')[1].split('.')[0].split(':')
     startValue = int(startTime[0])*60*60 + int(startTime[1])*60 + int(startTime[-1]) + int(time_gyro_1[0].split('.')[1])*0.001
     list_time_gyro_1 = []
     for tm in time_gyro_1:
        va = tm.split(' ')[-1].split('.')[0].split(':')
        total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
        list_time_gyro_1.append(total - startValue)
with open('../GA1/GAcceleratorFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         acc_x_1.append(float(value[0]))
         acc_y_1.append(float(value[1]))
         acc_z_1.append(float(value[2]))
         time_acc_1.append(value[3])
     startTime = time_acc_1[0].split(' ')[1].split('.')[0].split(':')
     startValue = int(startTime[0])*60*60 + int(startTime[1])*60 + int(startTime[-1]) + int(time_acc_1[0].split('.')[1])*0.001
     list_time_acc_1 = []
     for tm in time_acc_1:
        va = tm.split(' ')[-1].split('.')[0].split(':')
        total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
        list_time_acc_1.append(total - startValue)


# on the bad wall
gyro_x_2 = []
gyro_y_2 = []
gyro_z_2 = []

acc_x_2 = []
acc_y_2 = []
acc_z_2 = []
with open('../GA2/GyroscopeFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         gyro_x_2.append(float(value[0]))
         gyro_y_2.append(float(value[1]))
         gyro_z_2.append(float(value[2]))
        
with open('../GA2/GAcceleratorFrequency'+pattern+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
         value = line.rstrip().split(',')
         acc_x_2.append(float(value[0]))
         acc_y_2.append(float(value[1]))
         acc_z_2.append(float(value[2]))

# gyroscope data plot
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(list_time_gyro_0, gyro_z_0, label='Healthy wall')
plt.plot(list_time_gyro_1, gyro_z_1, label='Wall with cracks')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('time(s)', fontsize=18)
plt.ylabel('Radians/s', fontsize=18)
plt.ylim([-1, 1])
fig.subplots_adjust(left=0.15)
#plt.savefig('gyro_bad_wall.pdf')
plt.show()

