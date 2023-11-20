import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score, classification_report, confusion_matrix
import sklearn.metrics
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from statistics import mean
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import sklearn.model_selection as model_selection
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             f1_score, ConfusionMatrixDisplay,
                             classification_report)


#Membaca data
dataframe = pd.read_excel("metadata.xlsx")

data = dataframe[['image','format',
                  'mode','shape','class']]

print(" Data Awal ".center(75,"="))
print(data)
print("============================================================")

#Pengecekan missing value
print(" Pengecekan missing value ".center(75,"="))
print(data.isnull().sum())
print("============================================================")

#Grouping yang dibagi menjadi dua
print("GROUPING VARIABEL".center(75,"="))
X=data.iloc[:,0:4].values
y=data.iloc[:,4].values
#Ubah User ID dan Gender menjadi angka sederhana
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:, 0])
X[:,1] = le.fit_transform(X[:, 1])
X[:,2] = le.fit_transform(X[:, 2])
X[:,3] = le.fit_transform(X[:, 3])
data['class'] = le.fit_transform(data['class'])

print(" Data Variabel ".center(75,"="))
print(X)
print(" Data Kelas ".center(75,"="))
print(y)
print("============================================================")


# Penanganan Data Missing Value
print()
print("-----Penanganan Missing Value dengan Menghapus-----")
data = data.dropna()
print(data.isna().sum())
print()
def handle_outliers_zscore(data, column):
    z_scores = (data[column] - data[column].mean()) / data[column].std()
    outliers = data[np.abs(z_scores) > 3]
    data[column] = np.where(np.abs(z_scores) > 3, data[column].mean(), data[column])
    return outliers

# mendeteksi outlier setiap kolom
outliers_image = handle_outliers_zscore(data, 'image')
outliers_format = handle_outliers_zscore(data, 'format')
outliers_mode = handle_outliers_zscore(data, 'mode')
outliers_shape = handle_outliers_zscore(data, 'shape')

print("Penanganan Outlier dengan mengganti dengan mean:")
print(data)

# Min-Max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data[['image', 'format', 'mode', 'shape']])

print("Normalized Data:")
print(normalized_data)

#Pembagian training dan testing
print("SPLITTING DATA 20-80".center(75,"="))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print("instance variabel data training".center(75,"="))
print(X_train)
print("instance kelas data training".center(75,"="))
print(y_train)
print("instance variabel data testing".center(75,"="))
print(X_test)
print("instance kelas data testing".center(75,"="))
print(y_test)
print("============================================================")


#pemodelan naive bayes 
print("PEMODELAN DENGAN NAIVE BAYES".center(75,"="))
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print("instance prediksi naive bayes:")
print(Y_pred)

#Perhitungan confusion matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT NAIVE BAYES'.center(75,'='))
#Mendapat Akurasi
accuracy = accuracy_score(y_test, Y_pred)
# Mendapat Akurasi
precision = precision_score(y_test, Y_pred, pos_label='normal')
# Menampilkan recision    recall  f1-score   support
print(classification_report(y_test, Y_pred))
cm = confusion_matrix(y_test, Y_pred)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0
TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
total = TN + FN + TP + FP
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100
    
print('Akurasi : ', accuracy * 100, "%")
print('Sensitivity : ' + str(sens))
print('Specificity : ' + str(spec))    
print('Precision : ' + str(precision))
print("============================================================")
print()

#Menampilkan Confusion Matrix
cm_display=ConfusionMatrixDisplay(confusion_matrix=cm)
print('Confusion matrix for Naive Bayes\n',cm)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, Y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("============================================================")
print()

# Pemodelan Decision Tree
print("PEMODELAN DENGAN DECISION TREE".center(75, "="))
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred_dt = decision_tree.predict(X_test)

# Prediksi decision tree
print("Instance prediksi Decision Tree:")
print(Y_pred_dt)
print("============================================================")
print()

# Perhitungan confusion matrix
cm_dt = confusion_matrix(y_test, Y_pred_dt)
print('CLASSIFICATION REPORT DECISION TREE'.center(75, '='))

# Perhitungan akurasi
accuracy_dt = round(accuracy_score(y_test, Y_pred_dt) * 100, 2)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

# Mendapatkan akurasi
accuracy_dt = accuracy_score(y_test, Y_pred_dt)
precision_dt = precision_score(y_test, Y_pred_dt, pos_label='normal')

# Menampilkan recision   recall   f1-score   support
print(classification_report(y_test, Y_pred_dt))
TN_dt = cm_dt[1][1] * 1.0
FN_dt = cm_dt[1][0] * 1.0
TP_dt = cm_dt[0][0] * 1.0
FP_dt = cm_dt[0][1] * 1.0
total_dt = TN_dt + FN_dt + TP_dt + FP_dt
sens_dt = TN_dt / (TN_dt + FP_dt) * 100
spec_dt = TP_dt / (TP_dt + FN_dt) * 100

print('Akurasi : ', accuracy_dt * 100, "%")
print('Sensitivity : ' + str(sens_dt))
print('Specificity : ' + str(spec_dt))
print('Precision : ' + str(precision_dt))
print("============================================================")
print()

# Menampilkan Confusion Matrix Decision Tree
cm_display_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
print('Confusion matrix for Decision Tree\n', cm_dt)
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, Y_pred_dt), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("============================================================")

# Pemodelan Random Forest
print("PEMODELAN DENGAN RANDOM FOREST".center(75, "="))
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
Y_pred_rf = random_forest.predict(X_test)

# Prediksi Random Forest
print("Instance prediksi Random Forest:")
print(Y_pred_rf)

# Perhitungan confusion matrix
cm_rf = confusion_matrix(y_test, Y_pred_rf)
print('CLASSIFICATION REPORT RANDOM FOREST'.center(75, '='))

# Perhitungan akurasi
accuracy_rf = round(accuracy_score(y_test, Y_pred_rf) * 100, 2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

# Mendapatkan akurasi
accuracy_rf = accuracy_score(y_test, Y_pred_rf)
precision_rf = precision_score(y_test, Y_pred_rf, pos_label='normal')

# Menampilkan recision   recall   f1-score   support
print(classification_report(y_test, Y_pred_rf))
TN_rf = cm_rf[1][1] * 1.0
FN_rf = cm_rf[1][0] * 1.0
TP_rf = cm_rf[0][0] * 1.0
FP_rf = cm_rf[0][1] * 1.0
total_rf = TN_rf + FN_rf + TP_rf + FP_rf
sens_rf = TN_rf / (TN_rf + FP_rf) * 100
spec_rf = TP_rf / (TP_rf + FN_rf) * 100

print('Akurasi : ', accuracy_rf * 100, "%")
print('Sensitivity : ' + str(sens_rf))
print('Specificity : ' + str(spec_rf))
print('Precision : ' + str(precision_rf))
print("============================================================")
print()

# Menampilkan Confusion Matrix Random forest
cm_display_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
print('Confusion matrix for Random Forest\n', cm_rf)
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, Y_pred_rf), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("============================================================")
