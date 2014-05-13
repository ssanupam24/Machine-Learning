# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 1:08:17 2014

@author: anupam
"""
print(__doc__)

import numpy as np

# Create a random dataset
#rng = np.random.RandomState(1)
#X = np.sort(5 * rng.rand(80, 1), axis=0)
#y = np.sin(X).ravel()
#y[::5] += 3 * (0.5 - rng.rand(16))
f = open("breast-cancer-wisconsin.data")
x = []
y = []
z = []
count = 0
temp = []
valid = []
mis = 0
'''for line in f:
    y = line.split(",")
    x.append(y[10])

print(x)'''  
import csv
#c = csv.writer(open("MYFILE.csv", "wb"))
with open('breast-cancer-wisconsin.data', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        #print len(row)
        if len(row) == 11 and "?" not in row :
            x.append(row[1:10])
            y.append(int(row[10]))
for i in x:
    for j in i:
        temp.append(int(j))
    z.append(temp)
    temp = []
########################################################################
#NuSVM classifier
from sklearn.svm import NuSVC
clf = NuSVC()
clf.fit(z[1:200], y[1:200])
valid = clf.predict(z[201:698])      
for i in valid:
    if i != y[count+201]:
        mis+=1
    count+=1
print("NuSVM misclassification rate is")
print(float(float(mis)/498) * 100)
#########################################################################
#Random Forest
from sklearn.ensemble import RandomForestClassifier
mis = 0
count=0
clf1 = RandomForestClassifier(n_estimators=10)
clf1.fit(z[1:200], y[1:200])
RandomForestClassifier(n_estimators=10, max_depth=None,
                       min_samples_split=1, random_state=0)
valid1 = clf1.predict(z[201:698])
for i in valid1:
    if i != y[count+201]:
        mis+=1
    count+=1
print("Random forest misclassification rate is")
print(float(float(mis)/498) * 100)
#########################################################################
#Decision Trees
from sklearn.tree import DecisionTreeClassifier
mis = 0
count=0
clf3 = DecisionTreeClassifier()
clf3.fit(z[1:200], y[1:200])
DecisionTreeClassifier(max_depth=None,min_samples_split=1, random_state=0)
valid3 = clf3.predict(z[201:698])
for i in valid3:
    if i != y[count+201]:
        mis+=1
    count+=1
print("Decision trees misclassification rate is")
print(float(float(mis)/498) * 100)
#########################################################################
#Decision Trees AdaBoost
from sklearn.ensemble import AdaBoostClassifier
mis = 0
count=0
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=100)
bdt.fit(z[1:200], y[1:200])
valid2 = bdt.predict(z[201:698])
for i in valid2:
    if i != y[count+201]:
        mis+=1
    count+=1
print("AdaBoost Decision trees misclassification rate is")
print(float(float(mis)/498) * 100)
#########################################################################
#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
mis = 0
count=0
clf4 = GradientBoostingClassifier(n_estimators=100)
clf4.fit(z[1:200], y[1:200])
GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
                           max_depth=1, random_state=0)
valid4 = clf4.predict(z[201:698])
for i in valid4:
    if i != y[count+201]:
        mis+=1
    count+=1
print("GradientBoostingClassifier misclassification rate is")
print(float(float(mis)/498) * 100)
#########################################################################
