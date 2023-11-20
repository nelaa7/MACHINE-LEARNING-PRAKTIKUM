import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# load data
iris = datasets.load_iris()
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns=iris['feature_names']+['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
del df['target']
print(df)

# memisahkan label dan parammeter
x=df.iloc[:,:-1]
y=df.iloc[:,4]
x_train.x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=100)

# pemodelan sv,
















