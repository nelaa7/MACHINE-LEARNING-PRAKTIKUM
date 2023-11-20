import pandas as pd

# Membaca data dari file CSV
data = pd.read_csv('diabetes3.csv')

# Menampilkan data
print(data)

insulin = data[['Insulin','Age']]
print(insulin)