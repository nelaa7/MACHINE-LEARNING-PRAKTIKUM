import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Import Data
df = pd.read_csv('Engg_College_Data.csv')
df.head() 

df.shape
df.describe() 

# cek null data
df.isnull().sum() 

# tingkatkan visualisasi data
plt.style.use('fivethirtyeight') 

# amati masing-masing fitur 1
plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ['Teaching','Fees','Placements','Internship','Infrastructure']:
    n += 1
    plt.subplot(1 , 5 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()

# Ploting untuk mencari relasi antara Age , Annual Income and Spending Score 2
plt.figure(1 , figsize = (15 , 7))
n = 0
for x in ['Teaching','Fees','Placements','Internship','Infrastructure']:
    for y in ['Teaching','Fees','Placements','Internship','Infrastructure']:
        n += 1
        plt.subplot(5 , 5 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show() 

# plot Age dan Annual Income 3 
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Placements',y = 'Internship',data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 ,
    label = gender)
    plt.xlabel('Placements'), plt.ylabel('Internship')
    plt.title('Placements vs Internship')
    plt.legend()
plt.show()

# rancang K-Means untuk spending score vs annual income
# Kmeans, menentukan jumlah kluster dengan elbow 4
X1 = df[['Placements','Internship']].iloc[: ,:].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10
    ,max_iter=300,
    random_state= 111) )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_) 

# plot elbow 4
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show() 

# bangun K-Means 5
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10
,max_iter=300,
tol=0.0001, random_state= 111 , algorithm='elkan') )
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_ 

# siapkan data untuk plot dan imshow 
labels2 = algorithm.labels_ 
centroids2 = algorithm.cluster_centers_
step = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min,
y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) # array diratakan 1D

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(Z1 , interpolation='nearest',
extent=(xx.min(), xx.max(), yy.min(), yy.max()),
cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')
plt.scatter( x = 'Placements' ,y = 'Internship' , data = df , c = labels2 , s = 200 )
plt.scatter(x = centroids2[: , 0] , y = centroids2[: , 1] , s = 300 , c ='red' , alpha = 0.5)
plt.ylabel('Internship') , plt.xlabel('Placements')
plt.show() 

# coba prediksi
data = [[15, 39],[15, 20], [20, 80]]
print(data)
print(algorithm.predict(data))