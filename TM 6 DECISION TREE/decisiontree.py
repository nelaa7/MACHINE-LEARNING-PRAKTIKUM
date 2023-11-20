import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#membaca data
dataframe = pd.read_excel('BlaBla.xlsx')

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
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print("data variabel".center(75,"="))
print(X)
print("data kelas".center(75,"="))
print(y)
print("============================================================")

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

#pemodelan pemodelan decision tree
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

#prediksi decision tree
print("instance prediksi decision tree: ")
Y_pred = decision_tree.predict(X_test)
print(Y_pred)
print("============================================================")
print()

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

# Visualize the decision tree
plt.figure(figsize=(12, 8))
feature_names = data.columns[0:13].tolist()  # Convert Index object to a list of strings
plot_tree(decision_tree, filled=True, feature_names=feature_names)
plt.show()


print("============================================================")
print()

#COBA INPUT
print("CONTOH INPUT".center(75, '='))
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

predtest = decision_tree.predict(test)

if predtest==1:
    print("Pasien Positive ")
else:
    print("Pasien Negative ")