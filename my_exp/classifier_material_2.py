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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree

# classification using mfcc
good_file_baker_hall_1=open('./baker_hall/good_wall_mfcc.pkl', 'rb')
good_file_baker_hall_2=open('./baker_hall/TestData_1/good_wall_mfcc.pkl', 'rb')
bad_file_baker_hall_3=open('./baker_hall/bad_wall_mfcc.pkl', 'rb')
bad_file_baker_hall_4=open('./baker_hall/TestData_1/bad_wall_mfcc.pkl', 'rb')

good_file_office_1=open('./office_wall/good_wall_mfcc.pkl', 'rb')
good_file_office_2=open('./office_wall/TestData_1/good_wall_mfcc.pkl', 'rb')
bad_file_office_3=open('./office_wall/bad_wall_mfcc.pkl', 'rb')
bad_file_office_4=open('./office_wall/TestData_1/bad_wall_mfcc.pkl', 'rb')

good_file_baker_brics_1=open('./baker_brics/good_wall_mfcc.pkl', 'rb')
good_file_baker_brics_2=open('./baker_brics/TestData_1/good_wall_mfcc.pkl', 'rb')
bad_file_baker_brics_3=open('./baker_brics/bad_wall_mfcc.pkl', 'rb')
bad_file_baker_brics_4=open('./baker_brics/TestData_1/bad_wall_mfcc.pkl', 'rb')


hall_1 = pickle.load(good_file_baker_hall_1)
hall_2 = pickle.load(good_file_baker_hall_2)
hall_3 = pickle.load(bad_file_baker_hall_3)
hall_4 = pickle.load(bad_file_baker_hall_4)

office_1 = pickle.load(good_file_office_1)
office_2 = pickle.load(good_file_office_2)
office_3 = pickle.load(bad_file_office_3)
office_4 = pickle.load(bad_file_office_4)

brics_1 = pickle.load(good_file_baker_brics_1)
brics_2 = pickle.load(good_file_baker_brics_2)
brics_3 = pickle.load(bad_file_baker_brics_3)
brics_4 = pickle.load(bad_file_baker_brics_4)

hall = np.concatenate([hall_1[:,:,0:1,:,:], hall_2[:,:,0:1,:,:], hall_3[:,:,0:1,:,:], hall_4[:,:,0:1,:,:]], axis=0)
hall_re = np.reshape(hall, [hall.shape[0], hall.shape[1]*hall.shape[2]*hall.shape[3]*hall.shape[4]])

office = np.concatenate([office_1[:,:,0:1,:,:], office_2[:,:,0:1,:,:], office_3[:,:,0:1,:,:], office_4[:,:,0:1,:,:]], axis=0)
office_re = np.reshape(office,[office.shape[0], office.shape[1]*office.shape[2]*office.shape[3]*office.shape[4]])

brics = np.concatenate([brics_1[:,:,0:1,:,:], brics_2[:,:,0:1,:,:], brics_3[:,:,0:1,:,:], brics_4[:,:,0:1,:,:]], axis=0)
brics_re = np.reshape(brics, [brics.shape[0], brics.shape[1]*brics.shape[2]*brics.shape[3]*brics.shape[4]])


hall_len = hall_re.shape[0]
hall_y = np.ones((hall_len, 1), dtype=int)

office_len = office_re.shape[0]
office_y = 2*np.ones((office_len, 1), dtype=int)

brics_len = brics_re.shape[0]
brics_y = 3*np.ones((brics_len, 1), dtype=int)


#normalization
scaler_hall = MinMaxScaler()
scaler_hall.fit(hall_re)
hall_re_norm = scaler_hall.transform(hall_re)

scaler_office = MinMaxScaler()
scaler_office.fit(office_re)
office_re_norm = scaler_office.transform(office_re)

scaler_brics = MinMaxScaler()
scaler_brics.fit(brics_re)
brics_re_norm = scaler_brics.transform(brics_re)


data_x = np.concatenate([hall_re_norm, office_re_norm, brics_re_norm], axis=0)
data_y = np.concatenate([hall_y, office_y, brics_y], axis=0)


#data split and start to train the classifier
rng = np.random.RandomState(42)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=rng, shuffle=True)
"""
# svm classifier
clf = SVC(C=0.0005, kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# logistic classifier
clf = LogisticRegression(C=0.00005, random_state=rng).fit(x_train, y_train)
y_pred = clf.predict(x_test)
# random forest
clf = RandomForestClassifier(max_depth=2, n_estimators=10, max_features=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
"""
# knearest
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('precision, recall and f1 score:', precision_recall_fscore_support(y_test, y_pred, average='weighted'))
print('accuracy is %f:' % accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm = cm/cm.astype(np.float).sum(axis=1)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='.2f', ax=ax, annot_kws={"size":18})
ax.set_xlabel('Predicted', fontsize=18)
ax.set_ylabel('Actual', fontsize=18)
ax.xaxis.set_ticklabels(['Concrete', 'Wooden', 'Brics'], fontsize=18)
ax.yaxis.set_ticklabels(['Concrete', 'Wooden', 'Brics'], fontsize=18)
#plt.savefig('confusion_matrix_material.pdf')
plt.show()
