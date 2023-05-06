import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder


#Pre_processing.py File
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X


df = pd.read_csv('fifa19.csv')

name = df['Name']
cross = df['Crossing']
finishing = df['Finishing']
head = df['HeadingAccuracy']
shortPass = df['ShortPassing']
volleys = df['Volleys']

price = df['Value']

plt.scatter(cross,price, marker='x', c= 'r',label ='cross')
plt.scatter(finishing,price, marker='o', c= 'b',label ='finish')
plt.scatter(head,price, marker='x', c= 'y',label ='Header')


plt.xlabel('Crossing')
plt.ylabel('Value')
plt.title('Player Statistics')

#plt.show()









print(cross.max())
print(finishing.max())
print(head.max())
print(shortPass.max())
print(volleys.max())