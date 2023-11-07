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


good_file=open('good_wall.pkl', 'rb')
good_acc_gyro=pickle.load(good_file)
good_len = good_acc_gyro.shape[0]
good_y = np.ones((good_len, 1), dtype=int)

scaler_good = MinMaxScaler()
scaler_good.fit(good_acc_gyro)
good_acc_gyro = scaler_good.transform(good_acc_gyro)


bad_file=open('bad_wall.pkl', 'rb')
bad_acc_gyro=pickle.load(bad_file)
bad_len = bad_acc_gyro.shape[0]
bad_y = np.zeros((bad_len, 1), dtype=int)

scaler_bad = MinMaxScaler()
scaler_bad.fit(bad_acc_gyro)
bad_acc_gyro = scaler_bad.transform(bad_acc_gyro)


data_x = np.concatenate([good_acc_gyro, bad_acc_gyro], axis=0)
data_y = np.concatenate([good_y, bad_y], axis=0)


rng = np.random.RandomState(42)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=rng, shuffle=True)
# svm classifier
"""
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# logistic classifier
clf = LogisticRegression(random_state=rng).fit(x_train, y_train)
y_pred = clf.predict(x_test)
# decision tree classifier
clf = DecisionTreeClassifier(random_state=rng).fit(x_train, y_train)
y_pred = clf.predict(x_test)
"""
# k-nearest neighbors classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)


# roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
print("roc auc score: %f" % auc_score)
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr, label='ROC')
plt.plot(fpr, fpr, label='Reference line')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
#plt.savefig('roc_cur.pdf')
plt.show()

# pr curve
fig, ax = plt.subplots(figsize=(8,6))
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)
print("pr auc score %f" % pr_auc)
plt.plot(recall, precision, label='PR')
plt.plot([0, 1], [0.5, 0.5], label='Reference line')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
#plt.savefig('PR.pdf')
plt.show()

#confusion matrix
fig, ax = plt.subplots(figsize=(8,6))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = 1.0*fp/(tn+fp) 
tpr = 1.0*tp/(tp+fn)
recall = 1.0*tp/(tp+fn)
precision = 1.0*tp/(tp+fp)
f1 = (2.0*precision*recall)/(precision+recall) 
accuracy = (tp+tn)/(tp+fn+tn+fp)*1.0
print("accuracy is %f\n precision is %f\n recall is %f\n f1-score is %f\n" % (accuracy, precision, recall, f1))
cm = [
      [tp/(tp+fn)*1.0, fn/(tp+fn)*1.0],
      [fp/(fp+tn)*1.0, tn/(tn+fp)*1.0]
]
sns.heatmap(cm, annot=True, fmt='.2f', ax=ax, annot_kws={"size":18})
ax.set_xlabel('Predicted', fontsize=18)
ax.set_ylabel('Actual', fontsize=18)
ax.xaxis.set_ticklabels(['Healthy wall', 'Corroded wall'], fontsize=18)
ax.yaxis.set_ticklabels(['Healthy wall', 'Corroded wall'], fontsize=18)
#plt.savefig('confusion_matrix.pdf')
plt.show()
