
import csv
import matplotlib.pyplot as plt
import numpy as np
import sklearn as svm
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn import preprocessing 


data = []
topData = []
bottomData = []
array = [0,10,11,12,13,15,16,17,18,19]
with open("bank-additional-full.csv") as f:
    reader = csv.reader(f, delimiter=';')
    next(reader) # skip header
    for r in reader:
        data.append(r)
data = np.array(data)
dataClass = data[:,20]

#normalize
for number in array:
    dataTemp = data[:,number]
    print np.amin(dataTemp.astype(float)), np.amax(dataTemp.astype(float))
    normalizer = preprocessing.Normalizer(norm ='l2')
    temp = np.reshape(dataTemp, (-1,1))
    normalized = preprocessing.normalize(temp)
    print np.amin(normalized.astype(float)), np.amax(normalized.astype(float))
   


# Categorical: 1, 2, 3, 4, 5, 6, 7, 8, 9, 14
# numerical: 0 10 11 12 13 15 16 17 18 19
# ['56' 'housemaid' 'married' 'basic.4y' 'no' 'no' 'no' 'telephone' 'may'
#  'mon' '261' '1' '999' '0' 'nonexistent' '1.1' '93.994' '-36.4' '4.857'
#  '5191' 'no']
# CHI2
#  temp = np.reshape(temp, (-1,1))
# enc = preprocessing.LabelEncoder()
# new_temp = enc.fit_transform(temp)
# new_temp = new_temp.reshape(-1,1)
#  tempChi = svm.feature_selection.chi2(new_temp, data[:,20])
# MUTUAL INFORMATION 
# tempChi = svm.feature_selection.mutual_info_classif(temp, data[:,20])
# print tempChi
# 
#ONE HOT 
# temp = np.reshape(dataTemp, (-1,1))
#     enc = preprocessing.LabelEncoder() 
#     new_temp = enc.fit_transform(temp) 
#     new_temp = new_temp.reshape(-1,1) 
# 
#     replaced = np.unique(new_temp,return_counts=True)[0]
#     counts= np.unique(new_temp,return_counts=True)[1]
# 
#     solved = enc.inverse_transform(replaced)
#     for i in range(0,len(solved)):
#         print solved[i], ':', counts[i] 
# 
#     print '\n'
#     store = np.empty(len(new_temp[0]))
#     increment =0
# 

# #BARCHART code
# print len(topData)
# for x in labels:
#     topValues= np.append(topValues,list(topData).count(x))
# 
# for x in labels:
#     bottomValues = np.append(bottomValues,list(bottomData).count(x))
# p1 = plt.bar(ind,bottomValues, color='red')
# p2 = plt.bar(ind,topValues, bottom=bottomValues, color='blue')
# plt.title('Housing')
# plt.xticks(ind,labels,rotation=60)
# plt.legend(('No','Yes'))
# 
# 
# for r1,r2 in zip(p1,p2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x()+r1.get_width()/2., h1+h2, '%s'% (h1+h2), ha = 'center', va='bottom')
# 
# plt.show()
# 
# HISTOGRAM CODE
# p1=plt.hist([bottomData,topData],10 )
# print len(dataTemp)
# bins = p1[1]
# #p2=plt.hist(topData+0.5,10, color='red', width=0.5)
# plt.legend(('No','Yes'))
# #plt.figure(figsize=(100,100))
# plt.title('Previous')
# plt.xticks( bins)
# i = 0
# for height in p1[0][0]:
#     plt.text(bins[i]+3, height+120, int(height), ha = 'center', va='bottom')
#     i = i+1
# i=0
# for height in p1[0][1]:
#     plt.text(bins[i]+6, height, int(height), ha = 'center', va='bottom')
#     i = i+1
# plt.show()
