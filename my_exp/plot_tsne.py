import numpy as np
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing, metrics
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, average_precision_score, auc
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.manifold import TSNE


good_file_1=open('./baker_hall/good_wall.pkl', 'rb')
good_file_11=open('./baker_hall/TestData_1/good_wall.pkl', 'rb')
good_file_2=open('./office_wall/good_wall.pkl', 'rb')
good_file_22=open('./office_wall/TestData_1/good_wall.pkl', 'rb')
good_file_3=open('./baker_brics/good_wall.pkl', 'rb')
good_file_33=open('./baker_brics/TestData_1/good_wall.pkl', 'rb')
good_acc_gyro_1=pickle.load(good_file_1)
good_acc_gyro_11=pickle.load(good_file_11)
good_acc_gyro_2=pickle.load(good_file_2)
good_acc_gyro_22=pickle.load(good_file_22)
good_acc_gyro_3=pickle.load(good_file_3)
good_acc_gyro_33=pickle.load(good_file_33)
good_acc_gyro = np.concatenate([good_acc_gyro_1, good_acc_gyro_11, good_acc_gyro_2, good_acc_gyro_22, good_acc_gyro_3, good_acc_gyro_33], axis=0)
good_len = good_acc_gyro.shape[0]
good_y = np.ones((good_len, 1), dtype=int)

bad_file_1=open('./baker_hall/bad_wall.pkl', 'rb')
bad_file_11=open('./baker_hall/TestData_1/bad_wall.pkl', 'rb')
bad_file_2=open('./office_wall/bad_wall.pkl', 'rb')
bad_file_22=open('./office_wall/TestData_1/bad_wall.pkl', 'rb')
bad_file_3=open('./baker_brics/bad_wall.pkl', 'rb')
bad_file_33=open('./baker_brics/TestData_1/bad_wall.pkl', 'rb')
bad_acc_gyro_1=pickle.load(bad_file_1)
bad_acc_gyro_11=pickle.load(bad_file_11)
bad_acc_gyro_2=pickle.load(bad_file_2)
bad_acc_gyro_22=pickle.load(bad_file_22)
bad_acc_gyro_3=pickle.load(bad_file_3)
bad_acc_gyro_33=pickle.load(bad_file_33)
bad_acc_gyro = np.concatenate([bad_acc_gyro_1, bad_acc_gyro_11, bad_acc_gyro_2, bad_acc_gyro_22, bad_acc_gyro_3, bad_acc_gyro_33], axis=0)
bad_len = bad_acc_gyro.shape[0]
bad_y = np.zeros((bad_len, 1), dtype=int)

#normalization
scaler_good = MinMaxScaler()
scaler_good.fit(good_acc_gyro)
good_acc_gyro = scaler_good.transform(good_acc_gyro)

#normalization
scaler_bad = MinMaxScaler()
scaler_bad.fit(bad_acc_gyro)
bad_acc_gyro = scaler_bad.transform(bad_acc_gyro)


good_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(good_acc_gyro)
bad_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(bad_acc_gyro)

fig, ax = plt.subplots(figsize=(8,6))

plt.scatter(good_embed[1:201,0], good_embed[1:201,1], label='Wall 1')
plt.scatter(bad_embed[1:201,0], bad_embed[1:201,1], label='Wall 2')

plt.scatter(good_embed[201:401,0], good_embed[201:401,1], label='Wall 3')
plt.scatter(bad_embed[201:401,0], bad_embed[201:401,1], label='Wall 4')

plt.scatter(good_embed[401:601,0], good_embed[401:601,1], label='Wall 5')
plt.scatter(bad_embed[401:601,0], bad_embed[401:601,1], label='Wall 6')
plt.legend(fontsize=15)
plt.xlabel('tSNE1', fontsize=18)
plt.ylabel('tSNE2', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim([-25, 25])
#plt.ylim([-25, 25])
plt.savefig('tsne_pattern.pdf')
#fig.subplots_adjust(left=0.15)
#fig.subplots_adjust(bottom=0.15)
plt.show()

