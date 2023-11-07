from datetime import datetime
from sqlite3 import Timestamp
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas


# read gyroscope data over x, y, z
#filepath = ['GyroscopeFrequency10.txt','GyroscopeFrequency20.txt','GyroscopeFrequency30.txt','GyroscopeFrequency40.txt','GyroscopeFrequency50.txt']
filepath = ['GyroscopeFrequency50.txt']
cnt = 0
for ls in filepath:
    x, y, z, time = [], [], [], []
    cnt+=1
    with open(ls, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = line.rstrip().split(',')
            x.append(float(value[0]))
            y.append(float(value[1]))
            z.append(float(value[2]))
            time.append(value[3])
    print(time[0])
    startTime = time[0].split(' ')[1].split('.')[0].split(':')
    startValue = int(startTime[0])*60*60 + int(startTime[1])*60 + int(startTime[-1]) + int(time[0].split('.')[1])*0.001
    # compute the relative time for the remaining samples
    listTime = []
    for tm in time:
        va = tm.split(' ')[-1].split('.')[0].split(':')
        total = int(va[0])*3600+int(va[1])*60+int(va[-1])+int(tm.split(' ')[-1].split('.')[1])*0.001
        listTime.append(total - startValue)

    fig, ax = plt.subplots(figsize=(8,6)) 
    plt.plot(listTime, x, label='Gyro.x')
    plt.plot(listTime, y, label='Gyro.y')
    plt.plot(listTime, z, label='Gyro.z')
    #plt.xticks(fontsize=5)
    #plt.yticks(fontsize=15)
    plt.xlabel('time(s)', fontsize=18)
    plt.ylabel('Radians/s', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.savefig(ls.split('.')[0]+'.pdf') # save the plotted figure in pdf file
    #plt.savefig(ls.split('.')[0]+'.png')
    plt.show()



