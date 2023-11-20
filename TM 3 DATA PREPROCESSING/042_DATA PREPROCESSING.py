import numpy as np
import pandas as pd
from statistics import mean
from sklearn import preprocessing

# Membaca Data
data = pd.read_excel("kidneydisease.xlsx")
# data = pd.read_csv('kidneydisease.csv')
print(data.head())

# Mengambil beberapa atribut / variabel
data1 = data.loc[:, ['blood_glucose_random', 'blood_urea', 'serum_creatinine']]
print(data1.head())

# MENDETEKSI DATA MISSING
print("Deteksi Missing Value")
print(data1.isna().sum())

# Penanganan Data Missing Value
## MENGHAPUS DATA MISSING VALUE
print("Penanganan Missing Value")
data_cleaned = data1.dropna()
print("Data tanpa missing value")
print(data_cleaned)

# Penanganan Data Missing Value
## MENGGANTI DATA MISSING VALUE DENGAN MEAN
print("Penanganan Missing Value 2")
data1['blood_glucose_random'].fillna(data1['blood_glucose_random'].mean(), inplace=True)
data1['blood_urea'].fillna(data1['blood_urea'].mean(), inplace=True)
data1['serum_creatinine'].fillna(data1['serum_creatinine'].mean(), inplace=True)
print("Missing data pada blood glucose =", data1['blood_glucose_random'].isna().sum())
print("Missing data pada blood urea =", data1['blood_urea'].isna().sum())
print("Missing data pada serum creatinine =", data1['serum_creatinine'].isna().sum())

# Menampilkan nilai mean setelah penanganan missing value
mean_blood_glucose_random = data1['blood_glucose_random'].mean()
mean_blood_urea = data1['blood_urea'].mean()
mean_serum_creatinine = data1['serum_creatinine'].mean()

print("Mean untuk 'blood_glucose_random':", mean_blood_glucose_random)
print("Mean untuk 'blood_urea':", mean_blood_urea)
print("Mean untuk 'serum_creatinine':", mean_serum_creatinine)

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
outlier1 = detect_outlier(data1['blood_glucose_random'])
print("Outlier kolom blood_glucose_random : ", outlier1)
print("Banyak outlier blood_glucose_random : ", len(outlier1))
print()

outlier2 = detect_outlier(data1['blood_urea'])
print("Outlier kolom blood_urea : ", outlier2)
print("Banyak outlier blood_urea : ", len(outlier2))

outlier3 = detect_outlier(data1['serum_creatinine'])
print("Outlier kolom serum_creatinine : ", outlier3)
print("Banyak outlier serum_creatinine : ", len(outlier3))
print()

# Penanganan Outlier
variabel = ['blood_glucose_random', 'blood_urea', 'serum_creatinine']
for var in variabel:
    outlier_datapoints = detect_outlier(data1[var])
    print("Outlier ", var, " = ", outlier_datapoints)

# Penanganan Outlier untuk Mengganti outlier dengan nilai rata-rata (mean)
for var in variabel:
    outlier_datapoints = detect_outlier(data1[var])
    rata = mean(data1[var])
    data1[var] = data1[var].replace(outlier_datapoints, rata)

# Menampilkan data setelah penanganan outlier
print("Data setelah penanganan outlier:")
print(data1) 