import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix #menetentukan betul salah nya berapa
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier 
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import sklearn.model_selection as model_selection #untuk decision tree
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             f1_score, ConfusionMatrixDisplay,
                             classification_report)

pd.options.mode.chained_assignment = None 

#membaca data
dataframe = pd.read_excel(r"D:\CAMPUS\SEMESTER 5\MACHINE LEARNING PRAKTIKUM\TM 5 NAIVE BAYES\BlaBla.xlsx")

data=dataframe[['A','B','C',
                'D','E','F',
                'G','H',
                'I','J',
                'K','L',
                'M','N']]

print("data awal".center(75,"="))
print(data)
print("============================================================")

#pengecekan missing value
print("pengecekan missing value".center(75,"="))
print(data.isnull().sum())
print("============================================================")


#grouping yang dibagi menjadi dua
print("GROUPING VARIABEL".center(75,"="))
X=data.iloc[:,0:13].values
y=data.iloc[:,13].values
print("data variabel".center(75,"="))
print(X)
print("data kelas".center(75,"="))
print(y)
print("============================================================")

#pembagian training 80% dan testing 20%

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

#pemodelan naive bayes 
print("PEMODELAN DENGAN NAIVE BAYES".center(75,"="))
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print("instance prediksi naive bayes:")
print(Y_pred)
print("=====================================================")
#perhitungan confusion matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT NAIVE BAYES'.center(75,'='))

#Mendapat Akurasi
accuracy = accuracy_score(y_test, Y_pred)
# Mendapat Akurasi
precision = precision_score(y_test, Y_pred)
# Menampilkan recision    recall  f1-score   support
print(classification_report(y_test, Y_pred))
    
cm = confusion_matrix(y_test, Y_pred)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0
TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
total = TN + FN + TP + FP
sens = TN / (TN + FP) * 100 #precision
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


#COBA INPUT
A = int(input("Umur Pasien = "))
print("Isi Jenis kelamin dengan 0 jika Perempuan dan dan 1 jika Laki-Laki")
B = input("Jenis Kelamin Pasien = ")
print("Isi Y jika mengalami dan N jika tidak") 
C = input("Apakah pasien mengalami C? = ")
D = input("Apakah pasien mengalami D? = ")
E = input("Apakah pasien mengalami E? = ")
F = input("Apakah pasien mengalami F? = ")
G = input("Apakah pasien mengalami G? = ")
H = input("Apakah pasien mengalami H? = ")
I = input("Apakah pasien mengalami I? = ")
J = input("Apakah pasien mengalami J? = ")
K = input("Apakah pasien mengalami K? = ")
L = input("Apakah pasien mengalami L? = ")
M = input("Apakah M? = ")

umur_k = 0
A_k = 0
B_k = 0

if A<21:
   A_k=1
if A>20 and A<31:     
   A_k=2
if A>30 and A<41:
   A_k=3
if A>40 and A<51:
   A_k=4
   
if A>50:
   A_k=5
print("kode umur pasien adalah",A_k)

if B=="P":
   B_k=1
else:
   B_k=0

if C=="Y":
   C=1
else:
   C=0

if D=="Y":
    D=1
else:
   D=0

if E=="Y":
   E=1
else:
   E=0

if F=="Y":
   F=1
else:
   F=0

if G=="Y":
   G=1
else:
   G=0

if H=="Y":
   H=1
else:
   H=0

if I=="Y":
   I=1
else:
   I=0

if J=="Y":
   J=1
else:
   J=0

if K=="Y":
   K=1
else:
   K=0

if L=="Y":
   L=1
else:
   L=0

if M=="Y":
   M=1
else:
   M=0

Train = [A_k,B_k,C,D,E,F,G,
         H,I,J,K,L,M]
print(Train)

test = pd.DataFrame(Train).T

predtest = gaussian.predict(test)

if predtest==1:
    print("Pasien Positive ")
else:
    print("Pasien Negative ")

