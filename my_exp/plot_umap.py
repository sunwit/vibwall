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
import umap
import umap.plot
from umap import UMAP
import plotly.express as px


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

#good_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(good_acc_gyro)
#bad_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(bad_acc_gyro)

print(good_acc_gyro.shape)
x = np.concatenate([good_acc_gyro, bad_acc_gyro], axis=0)#[:,1:17]
#y = np.concatenate([good_y, bad_y], axis=0).flatten()
good_l = good_acc_gyro.shape[0]
total_l = x.shape[0]



fig, ax = plt.subplots(figsize=(8,6))

reducer = umap.UMAP()
embed = reducer.fit_transform(x)
plt.scatter(embed[1:good_l, 0], embed[1:good_l,1], label='Wall without cracks')
plt.scatter(embed[good_l:total_l, 0], embed[good_l:total_l,1], label='Wall with cracks')
plt.xlabel('UMAP dimension 1', fontsize=18)
plt.ylabel('UMAP dimension 2', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot_umap.pdf')
fig.subplots_adjust(left=0.15)
#fig.subplots_adjust(bottom=0.15)
plt.show()





#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.xlim([-25, 25])
#plt.ylim([-25, 25])
#plt.savefig('tsne_three_pattern.pdf')
#fig.subplots_adjust(left=0.15)
#fig.subplots_adjust(bottom=0.15)
plt.show()
