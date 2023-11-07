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
from sklearn.manifold import TSNE


file_1=open('./baker_hall/good_wall.pkl', 'rb')
file_11=open('./baker_hall/TestData_1/good_wall_mfcc.pkl', 'rb')
file_2=open('./baker_hall/bad_wall.pkl', 'rb')
file_22=open('./baker_hall/TestData_1/bad_wall.pkl', 'rb')
acc_gyro_1=pickle.load(file_1)
acc_gyro_11=pickle.load(file_11)
acc_gyro_2 = pickle.load(file_2)
acc_gyro = np.concatenate([acc_gyro_1, acc_gyro_11, acc_gyro_2], axis=0)
good_len = acc_gyro.shape[0]
bakerHall_y = np.ones((good_len, 1), dtype=int)

#normalization
scaler_good = MinMaxScaler()
scaler_good.fit(acc_gyro)
bakerHall_acc_gyro = scaler_good.transform(acc_gyro)


file_1=open('./office_wall/good_wall.pkl', 'rb')
file_2=open('./office_wall/bad_wall.pkl', 'rb')
acc_gyro_1=pickle.load(file_1)
acc_gyro_2=pickle.load(file_2)
acc_gyro = np.concatenate([acc_gyro_1, acc_gyro_2], axis=0)
bad_len = acc_gyro.shape[0]
office_y = np.zeros((bad_len, 1), dtype=int)

#normalization
scaler_bad = MinMaxScaler()
scaler_bad.fit(acc_gyro)
office_acc_gyro = scaler_bad.transform(acc_gyro)


file_1=open('./baker_brics/good_wall.pkl', 'rb')
file_2=open('./baker_brics/bad_wall.pkl', 'rb')
acc_gyro_1=pickle.load(file_1)
acc_gyro_2=pickle.load(file_2)
acc_gyro = np.concatenate([acc_gyro_1, acc_gyro_2], axis=0)
bad_len = acc_gyro.shape[0]
bakerBrics_y = 2*np.ones((bad_len, 1), dtype=int)

#normalization
scaler_bad = MinMaxScaler()
scaler_bad.fit(acc_gyro)
bakerBrics_acc_gyro = scaler_bad.transform(acc_gyro)


data_x = np.concatenate([bakerHall_acc_gyro, office_acc_gyro, bakerBrics_acc_gyro], axis=0)
data_y = np.concatenate([bakerHall_y, office_y, bakerBrics_y], axis=0)


rng = np.random.RandomState(42)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=rng, shuffle=True)
"""
# svm classifier
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# logistic classifier
clf = LogisticRegression(random_state=rng).fit(x_train, y_train)
y_pred = clf.predict(x_test)

# decision tree classifier
clf = DecisionTreeClassifier(random_state=rng).fit(x_train, y_train)
y_pred = clf.predict(x_test)

# k-nearest neighbors classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
"""
#print(accuracy_score(y_test, y_pred))

bakerHall_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(bakerHall_acc_gyro)
office_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(office_acc_gyro)
bakerBrics_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(bakerBrics_acc_gyro)


fig, ax = plt.subplots(figsize=(8,6))

plt.scatter(bakerHall_embed[:,0], bakerHall_embed[:,1], label='Concrete wall')
plt.scatter(office_embed[:,0], office_embed[:,1], label='Wooden wall')
plt.scatter(bakerBrics_embed[:,0], bakerBrics_embed[:,1], label='Brick wall')


plt.legend(fontsize=15)
plt.xlabel('tSNE1', fontsize=18)
plt.ylabel('tSNE2', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim([-25, 25])
#plt.ylim([-25, 25])
#plt.savefig('tsne_three_pattern.pdf')
#fig.subplots_adjust(left=0.15)
#fig.subplots_adjust(bottom=0.15)
plt.show()

