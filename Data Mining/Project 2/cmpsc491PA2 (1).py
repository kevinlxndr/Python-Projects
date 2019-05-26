
# coding: utf-8

# In[221]:


import csv
import matplotlib.pyplot as plt
import numpy as np
import sklearn as svm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score 
from sklearn import preprocessing 
from sklearn.utils import resample
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle


# In[250]:


data = []
dataClass = []
topData = []
bottomData = []
numericArray = [0,10,11,12,13,15,16,17,18,19]
categoricalArray = [1,2,3,4,5,6,7,8,9,14]
newcategoricalArray = [7,4,3,14,1]
selectData={}
    
with open("bank-additional-full.csv") as f:
    reader = csv.reader(f, delimiter=';')
    next(reader) # skip header
    for r in reader:
        data.append(r)
        
data = np.array(data)
resampled = resample(data,replace=True, n_samples =60000)
#data = resampled
dataClass = data[:,20]
label = preprocessing.LabelEncoder().fit(dataClass)
classes = label.transform(dataClass)

#41188
numeric = data[:,[0,10,11,12,13,15,16,17,18,19]].astype(float)
categorical = np.zeros(shape=(41188,10))

normalizer = preprocessing.Normalizer().fit(numeric)
#numeric = normalizer.transform(numeric)

for number in range(0,10):
    label = preprocessing.LabelEncoder().fit(data[:,newcategoricalArray[number]])
    categorical[:,number] = label.transform(data[:,newcategoricalArray[number]])
    
ohe = preprocessing.OneHotEncoder(sparse=False).fit(categorical)
categorical = ohe.transform(categorical)

final = np.append(numeric,categorical,axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(final,classes, test_size=0.2 )
print xtrain.shape, xtest.shape, ytest.shape, ytrain.shape
newclasses = ytest

logreg = LogisticRegression().fit(xtrain,ytrain)
#predict = cross_val_predict(logreg,final,classes,cv=5)
predict = logreg.predict(xtest)
print accuracy_score(newclasses,predict), precision_score(newclasses,predict), recall_score(newclasses,predict), f1_score(newclasses,predict)


dt = tree.DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
predict = dt.predict(xtest)
print accuracy_score(newclasses,predict), precision_score(newclasses,predict), recall_score(newclasses,predict), f1_score(newclasses,predict)

mlp = MLPClassifier()
mlp.fit(xtrain,ytrain)
predict = mlp.predict(xtest)
print accuracy_score(newclasses,predict), precision_score(newclasses,predict), recall_score(newclasses,predict), f1_score(newclasses,predict)

