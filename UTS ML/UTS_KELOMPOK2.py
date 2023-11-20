import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import sklearn.metrics
from statistics import mean
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix #menetentukan betul salah nya berapa
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score 
from collections import Counter
import sklearn.model_selection as model_selection #untuk decision tree
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             f1_score, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# from scipy.stats import gaussian


#membaca data
# dataframe = pd.read_excel("lung_cancer.xlsx")
dataframe = pd.read_excel("lung_cancer_examples.xlsx")


data = dataframe[['Age', 'Smokes', 
                'AreaQ','Alkhol', 'Result']]

# print("data awal".center(75,"="))
# print(data)
# X=data.iloc[:,0:6].values
# le = LabelEncoder()
# X[:,0] = le.fit_transform(X[:, 0])
# X[:,1] = le.fit_transform(X[:, 1])
# X[:,2] = le.fit_transform(X[:, 2])
# X[:,3] = le.fit_transform(X[:, 3])
# X[:,4] = le.fit_transform(X[:, 4])
# X[:,5] = le.fit_transform(X[:, 5])
# data['Name'] = le.fit_transform(data['Name'])
# data['Surname'] = le.fit_transform(data['Surname'])
# print("============================================================")


# #Pengecekan Missing Value
print("pengecekan missing value".center(75,"="))
print(data.isnull().sum())
print("============================================================")

#Mengambil beberapa Atribut / variabel
data1 = data.loc[:, ['Age', 'Smokes', 'AreaQ']]
print(data1.head())


#Penanganan Data Missing Value
## MENGHAPUS DATA MISSING VALUE
print("Penanganan Missing Value 1")
Missing_Age = data1['Age'].isna().dropna()
Missing_Smokes = data1['Smokes'].isna().dropna()
Missing_AreaQ = data1['AreaQ'].isna().dropna()
print("Missing data pada Age = ", Missing_Age.isna().sum())
print("Missing data pada Smokes = ", Missing_Smokes.isna().sum())
print("Missing data pada AreaQ = ", Missing_AreaQ.isna().sum())

#MENGGANTI DATA MISSSING VALUE DENGAN MEAN
print("Penanganan Missing value 2")
RataAge = data1['Age'].mean
Missing_Age = data1['Age'].fillna(RataAge)
RataSmokes = data1['Smokes'].mean
Missing_Smokes = data1['Smokes'].fillna(RataSmokes)
RataAreaQ = data1['AreaQ'].mean
Missing_AreaQ = data1['AreaQ'].fillna(RataAreaQ)
print("Missing data pada Age =", Missing_Age.isna().sum())
print("Missing data pada Smokes =", Missing_Smokes.isna().sum())
print("Missing data pada AreaQ =", Missing_AreaQ.isna().sum())


##############################
# Step 2

# Mendeteksi Outlier
print("Deteksi Outlier")
outliers = []

def detect_outlier(data):
    threshold = 3
    mean_value = data.mean()
    std_dev = data.std()

    for x in data:
        z_score = (x - mean_value) / std_dev
        if np.abs(z_score) > threshold:
            outliers.append(x)
    return outliers


# Mencetak Outlier
print("============================================================")
outlier1 = detect_outlier(data['Age'])
print("Outlier kolom Age : ", outlier1)
print("Banyak outlier Age : ", len(outlier1))
print()

print("============================================================")
outlier2 = detect_outlier(data['Smokes'])
print("Outlier kolom Smokes : ", outlier2)
print("Banyak outlier Smokes : ", len(outlier2))
print()

print("============================================================")
outlier3 = detect_outlier(data['AreaQ'])
print("Outlier kolom AreaQ : ", outlier3)
print("Banyak outlier AreaQ : ", len(outlier3))
print()

#Penanganan Outlier
variabel = ['Age', 'Smokes', 'AreaQ']
for var in variabel:
    outlier_datapoints = detect_outlier(data1[var])
    print("Outlier ", var, "=", outlier_datapoints)
    rata = mean(data1[var])
    print("============================================================")
    print("Outlier ", var, "telah diganti menjadi mean :")
    data1[var] = data1[var].replace(outlier_datapoints, rata)
    print(data1)

##############################
# Step 3

#Mengambil beberapa atribut / variabel
data1 = data.loc[:, ['Age', 'Smokes', 'AreaQ']]
print(data1.head())

print()
#Normalisasi
#Feature Scaling or Standarddization
scaler1 = StandardScaler()
Normalisasi1 =scaler1.fit_transform(data1)
print("------ Hasil Feaature Scaling = ")
print(Normalisasi1)

print()
#Normalisasi z score
zscores = stats.zscore(data1, axis=1)
print("Hasil Z Score = ")
print(zscores) 
print("====================================")

##############################
# Step 4
#Naive Baiyest

# #grouping yang dibagi menjadi dua
print("GROUPING VARIABEL".center(75,"="))
X=data.iloc[:,0:4].values
y=data.iloc[:,4].values
print("data variabel".center(75,"="))
print(X)
print("data kelas".center(75,"="))
print(y)
print("============================================================")
print()

#pembagian training dan testing
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
print()

# #Pemodelan Naive Bayes 

# print("PEMODELAN DENGAN NAIVE BAYES".center(75,"="))
# gaussian = GaussianNB()
# gaussian.fit(X_train, y_train)
# Y_pred = gaussian.predict(X_test) 
# accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
# acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
# print("instance prediksi naive bayes:")
# print(Y_pred)
# #perhitungan confusion matrix
# cm = confusion_matrix(y_test, Y_pred)
# print('CLASSIFICATION REPORT NAIVE BAYES'.center(75,'='))

# #Mendapat Akurasi
# accuracy = accuracy_score(y_test, Y_pred)
# # Mendapat Akurasi
# precision = precision_score(y_test, Y_pred)
# # Menampilkan recision    recall  f1-score   support
# print(classification_report(y_test, Y_pred))
    
# cm = confusion_matrix(y_test, Y_pred)
# TN = cm[1][1] * 1.0
# FN = cm[1][0] * 1.0

# TP = cm[0][0] * 1.0
# FP = cm[0][1] * 1.0
# total = TN + FN + TP + FP
# sens = TN / (TN + FP) * 100
# spec = TP / (TP + FN) * 100
    
# print('Akurasi : ', accuracy * 100, "%")
# print('Sensitivity : ' + str(sens))
# print('Specificity : ' + str(spec))
    
# print('Precision : ' + str(precision))
# print("============================================================")
# print()

# #Menampilkan Confusion Matrix
# cm_display=ConfusionMatrixDisplay(confusion_matrix=cm)

# print('Confusion matrix for Naive Bayes\n',cm)
# f, ax = plt.subplots(figsize=(8,5))
# sns.heatmap(confusion_matrix(y_test, Y_pred), annot=True, fmt=".0f", ax=ax)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

##############################
# # Step 5
# Decision Tree
#pemodelan pemodelan decision tree
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

#prediksi decision tree
print("instance prediksi decision tree:")
print()

Y_pred = decision_tree.predict(X_test)

#prediksi akurasi
accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
print("Akurasi: ", accuracy, "%")


# Display classification report
print("CLASSIFICATION REPORT DECISION TREE".center(75, '='))
print(classification_report(y_test, Y_pred))

# Display confusion matrix
cm = confusion_matrix(y_test, Y_pred)
print("Confusion Matrix:")
print(cm)

# Visualisasi matriks kebingungan
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Visualize the decision tree
plt.figure(figsize=(12, 8))
feature_names = data.columns[0:13].tolist()  # Convert Index object to a list of strings
plot_tree(decision_tree, filled=True, feature_names=feature_names)
plt.show()


##############################
#Step 6
# Step Random Forest

# # Muat dataset (misalnya, dataset Iris)
# iris = load_iris()
# X, y = iris.data, iris.target

# # Pisahkan data menjadi data pelatihan dan data pengujian
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Inisialisasi model Random Forest
# rf_model = RandomForestClassifier(n_estimators=3, random_state=42)

# # Latih model pada data pelatihan
# rf_model.fit(X_train, y_train)

# # Lakukan prediksi pada data pengujian
# Y_pred = rf_model.predict(X_test)

# # # Hitung akurasi model pada data pengujian
# accuracy = accuracy_score(y_test, Y_pred)
# print(f'Akurasi Model: {accuracy}')

# #prediksi akurasi
# accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
# print("Akurasi: ", accuracy, "%")

# # Display confusion matrix
# cm = confusion_matrix(y_test, Y_pred)
# print("Confusion Matrix:")
# print(cm)

# # Visualisasi matriks kebingungan
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - Decision Random Forest Tree")
# plt.show()


# # Membuat dan mencetak pohon-pohon dalam model Random Forest
# for i, tree_in_forest in enumerate(rf_model.estimators_):
#     plt.figure(figsize=(12, 8))
#     feature_names = data.columns[0:4].tolist()
#     plot_tree(tree_in_forest, filled=True, feature_names=feature_names)
#     # plot_tree(tree_in_forest, feature_names=iris.feature_names, filled=True, class_names=iris.target_names)
#     plt.title(f"Decision Tree {i+1}")
#     plt.show()

# # #COBA INPUT
# print("Masukkan data uji".center(75,"="))
# # # # A = str (input("Nama = "))
# # # # B = str (input("Nama Belakang = "))
# C = int(input("Usia Pasien = "))
# print("------ Ukuran Pasien ------")
# D = int(input("input Smokes? = "))
# E = int(input("input AreaQ? = "))
# F = int(input("input Alkohol? = "))



# test_data =  [[ C, D, E, F]]
# pred_test = gaussian.predict(test_data)

# print()
# if pred_test == 1:
#     print("Pasien Mengalami Serangan Kanker Paru")
# else:
#     print("Pasien Normal Sehat ")


# TM 8 SVM 







