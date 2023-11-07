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


good_file_1=open('./good_wall_wall_mfcc.pkl', 'rb')
good_acc_gyro_1=pickle.load(good_file_1)

#good_acc_gyro_con = np.concatenate([good_acc_gyro_1[:,:,0:4,:,:], good_acc_gyro_11[:,:,0:4,:,:],  good_acc_gyro_2[:,:,0:4,:,:], good_acc_gyro_22[:,:,0:4,:,:], good_acc_gyro_3[:,:,0:4,:,:], good_acc_gyro_33[:,:,0:4,:,:]], axis=0)
good_acc_gyro_con = good_acc_gyro_1[:,:,0:5,:,:]
good_acc_gyro = np.reshape(good_acc_gyro_con, [good_acc_gyro_con.shape[0], good_acc_gyro_con.shape[1]*good_acc_gyro_con.shape[2]*good_acc_gyro_con.shape[3]*good_acc_gyro_con.shape[4]])

good_len = good_acc_gyro.shape[0]
good_y = np.ones((good_len, 1), dtype=int)


bad_file_1=open('./bad_wall_wall_mfcc.pkl', 'rb')
bad_acc_gyro_1=pickle.load(bad_file_1)
#bad_acc_gyro_con = np.concatenate([bad_acc_gyro_1[:,:,0:4,:,:], bad_acc_gyro_11[:,:,0:4,:,:], bad_acc_gyro_2[:,:,0:4,:,:], bad_acc_gyro_22[:,:,0:4,:,:], bad_acc_gyro_3[:,:,0:4,:,:], bad_acc_gyro_33[:,:,0:4,:,:]], axis=0)
bad_acc_gyro_con = bad_acc_gyro_1[:,:,0:5,:,:]

bad_acc_gyro = np.reshape(bad_acc_gyro_con, [bad_acc_gyro_con.shape[0], bad_acc_gyro_con.shape[1]*bad_acc_gyro_con.shape[2]*bad_acc_gyro_con.shape[3]*bad_acc_gyro_con.shape[4]])


#normalization
scaler_good = MinMaxScaler()
scaler_good.fit(good_acc_gyro)
good_acc_gyro = scaler_good.transform(good_acc_gyro)

#normalization
scaler_bad = MinMaxScaler()
scaler_bad.fit(bad_acc_gyro)
bad_acc_gyro = scaler_bad.transform(bad_acc_gyro)



x = np.concatenate([good_acc_gyro, bad_acc_gyro], axis=0)
good_l = good_acc_gyro.shape[0]
total_l = x.shape[0]



fig, ax = plt.subplots(figsize=(8,6))

reducer = umap.UMAP()
embed = reducer.fit_transform(x)

plt.scatter(embed[1:good_l, 0], embed[1:good_l,1], label='Wall without cracks')
plt.scatter(embed[good_l:total_l, 0], embed[good_l:total_l,1], label='Wall with cracks in depth of 1cm')

plt.xlabel('UMAP dimension 1', fontsize=18)
plt.ylabel('UMAP dimension 2', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-10, 25])
plt.ylim([-10, 25])
plt.legend(fontsize=18)
plt.savefig('plot_umap_mfcc_depth_1cm.pdf')
fig.subplots_adjust(left=0.15)
#fig.subplots_adjust(bottom=0.15)
plt.show()
