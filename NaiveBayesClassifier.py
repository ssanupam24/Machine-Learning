# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:56:44 2014


@author: anupam
"""

#import pdb
import random
#import matplotlib.pyplot as plt
#import numpy as np
#import math
#pdb.set_trace()
numbins=3

def print1 (result):
    for i in result:
        print i
        print "\n"   

def min(listall,i):
    minval=listall[0][i]
    for line in listall:
        if minval > line[i]:
            minval=line[i]
    return minval
    
def max(listall,i):
    maxval=listall[0][i]
    for line in listall:
        if maxval < line[i]:
            maxval=line[i]
    return maxval  
    
##This function set the count of the data in that bin,where w=min bin width minval=minimumvaluefeature
##and a is bin of corresponding feature
   
def set(traindata,a,feature,minval,w):
    ##rint "Inside="
    ##print1(a)
    for line in traindata:
        x=int(line[4])
        n=line[feature]
        y=int((float(n)-float(minval))/float(w))
        ##print a[x][y]
        ##print x
        ##print y
        a[x][y]=(int)((int)(a[x][y])+1)
    return a
        
        
 ## This function make the bin which have 3 row for 3 classifier  and Width denotes number of bin       
def makebin(bin,width):
    for i in xrange(3):
        bin.append([])
        for j in xrange(width):
            bin[i].append(1)    
    return bin
 

##Size=Number of feature for that classifier
## Totalsize= size of traindata
## i= which classifire we are talking about
## val[0]= feature value    
def probability(bin,size,i,val,minval,w):  
    j=int((float(val)-float(minval))/float(w))
    p=float(float(bin[i][j])/float(size))
    return p


##Size=Number of feature for that classifier
## Totalsize= sie of traindata
## i= which classifire we are talking about
## val[0]= feature value    
def probf(bin1,bin2,bin3,bin4,size,totalsize,i,val,min1,min2,min3,min4,intsz1,intsz2,intsz3,intsz4):
    p1=probability(bin1,size,i,val[0],min1,intsz1) 
    p2=probability(bin2,size,i,val[1],min2,intsz2)
    p3=probability(bin3,size,i,val[2],min3,intsz3) 
    p4=probability(bin4,size,i,val[3],min4,intsz4)   
    cp=(float)(float(size)/float(totalsize))
    print "Class is"
    print i    
    print "Class Probability"
    print cp
    ##p=cp*p1*p2*p3*p4  
    p=cp*(p1*p2*p3*p4)
    return p

def max1(f1,f2):
    if f1>f2:
        return f1
    else:
        return f2
    
    
def classifier(c1,c2,c3):
    f1=float(c1)
    f2=float(c2)
    f3=float(c3)
    f4=max1(f3,max1(f1,f2))

    if(abs(f4-f1)<0.005):
        return 0
    if(abs(f4-f2)<0.005):
         return 1
    if(abs(f4-f3)<0.005):
        return 2 
    
 ## For each set classify in 3 classifire   
## count = Count of classifire
## size = size of traindata
## values= Each object we need to classify
## feature bin=spLbin,spWbin,pLbin,pWbin,
## feature min val of bin=(spLbin,spWbin,pLbin,pWbin)min
 
 
def classify(testdata,spLbin,spWbin,pLbin,pWbin,count,size,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,s1,s2,s3,s4):
        result=[]
        misc=0
        for values in testdata:
            #print values
            p1=probf(spLbin,spWbin,pLbin,pWbin,count[0],size,0,values,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,s1,s2,s3,s4)
            p2=probf(spLbin,spWbin,pLbin,pWbin,count[1],size,1,values,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,s1,s2,s3,s4)
            p3=probf(spLbin,spWbin,pLbin,pWbin,count[2],size,2,values,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,s1,s2,s3,s4)
            finalclass=classifier(p1,p2,p3)
            res=[]
            res.append(finalclass)
            res.append(values[4])
            result.append(res)
            if int(finalclass)!=int(values[4]):
                misc=misc+1
                
            
        return result,misc

     

#############################################################################################             


## Listall store all the data in the list
listall=[]

## Open the file and store each line in the list
f = open('/home/anupam/Documents/ML/Assign_1/iris.txt', 'rw+')
for line in f:
        y=line.split()
        listall.append(y)
        

# Split into training  data
traindata1=[]
traindata2=[]
traindata3=[]

## split into Test data
testdata1=[]
testdata2=[]
testdata3=[]


### To make sure that there is no duplicate in training data 
i = 0
test1=[]
test2=[]
test3=[]

count1=[]
count2=[]
count3=[]


i=0
while i!=3:
    i=i+1
    count1.append(0)
    count2.append(0)
    count3.append(0)
    
###########################################################################################3
## Generate Training  data1
i=0
while i !=15:
    i=i+1
    while True:
        x=random.randint(0, 149)
        if x not in test1:
            count1[int(listall[x][4])]=count1[int(listall[x][4])]+1            
            test1.append(x)
            traindata1.append(listall[x])
            break
            
##print1 (traindata1) 
          
## Generate Test data 1

i=0
while i<len(listall):           
    if i not in test1:
        testdata1.append(listall[i])
    i=i+1          


##Generating Train Data2
i=0
while i !=45:
    i=i+1
    while True:
        x=random.randint(0, 149)
        if x not in test2:
            count2[int(listall[x][4])]=count2[int(listall[x][4])]+1            
            test2.append(x)
            traindata2.append(listall[x])
            break
            
##print len(traindata1) 
          
## Genreate Test data 1

i=0
while i<len(listall):           
    if i not in test2:
        testdata2.append(listall[i])
    i=i+1  



i=0
while i !=75:
    i=i+1
    while True:
        x=random.randint(0, 149)
        if x not in test3:
            count3[int(listall[x][4])]=count3[int(listall[x][4])]+1            
            test3.append(x)
            traindata3.append(listall[x])
            break
            
##print len(traindata1) 
          
## Genreate Test data 1

i=0
while i<len(listall):           
    if i not in test3:
        testdata3.append(listall[i])
    i=i+1  
     
###############################################################        

## calculate minimum and maximum of all the data    
sepallengthmin=min(listall,0)
sepalwidthmin=min(listall,1)
petallengthmin=min(listall,2)
petalwidthmin=min(listall,3)


sepallengthmax=max(listall,0)
sepalwidthmax=max(listall,1)
petallengthmax=max(listall,2)
petalwidthmax=max(listall,3)
################################################################
##print sepallengthmin
##print sepallengthmax

# Calculate Number of bins
intsz1=float(sepallengthmax)-float(sepallengthmin)
intsz1=float(intsz1)/float(numbins)
##print numbin1 

intsz2=float(sepalwidthmax)-float(sepalwidthmin)
intsz2=float(intsz2)/float(numbins)

intsz3=float(petallengthmax)-float(petallengthmin)
intsz3=float(intsz3)/float(numbins)

intsz4=float(petalwidthmax)-float(petalwidthmin)
#print petalwidthmax
#print petalwidthmin
intsz4=float(intsz4)/float(numbins)
#####################################################

spLbin = [] ## sepallength 
spWbin= []  ## sepalwidth
pLbin = []  ## petallength
pWbin= []   ## petalwidth
###########################################################
## Makebin for all feature 
spLbin=makebin(spLbin,numbins+1);
##print1 (spLbin)

##print1(spLbin)
spWbin=makebin(spWbin,numbins+1);
pLbin=makebin(pLbin,numbins+1);
pWbin=makebin(pWbin,numbins+1);


##############################################################

spLbin1=set(traindata1,spLbin,0,sepallengthmin,intsz1)
spWbin1=set(traindata1,spWbin,1,sepalwidthmin,intsz2)
pLbin1=set(traindata1,pLbin,2,petallengthmin,intsz3)
pWbin1=set(traindata1,pWbin,3,petalwidthmin,intsz4)

########################################################
#classify data based on testdata1

result1,misc1=classify(testdata1,spLbin1,spWbin1,pLbin1,pWbin1,count1,15,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,intsz1,intsz2,intsz3,intsz4)
#print intsz4
#print1 (pWbin1)
print "Misclassification percentage for Traindata1="
print float(misc1/135.0)*100
############################################################
#classify data basesd on testdata2
spLbin2=set(traindata2,spLbin,0,sepallengthmin,intsz1)
spWbin2=set(traindata2,spWbin,1,sepalwidthmin,intsz2)
pLbin2=set(traindata2,pLbin,2,petallengthmin,intsz3)
pWbin2=set(traindata2,pWbin,3,petalwidthmin,intsz4)
print "Test2***************************************"
result2,misc2=classify(testdata2,spLbin2,spWbin2,pLbin2,pWbin2,count2,45,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,intsz1,intsz2,intsz3,intsz4)
print "Misclassification percentage for Traindata2="
print (misc2/105.0)*100
#####################################################################################
print "Test3***************************************"
#classify data basesd on testdata2
spLbin3=set(traindata3,spLbin,0,sepallengthmin,intsz1)
spWbin3=set(traindata3,spWbin,1,sepalwidthmin,intsz2)
pLbin3=set(traindata3,pLbin,2,petallengthmin,intsz3)
pWbin3=set(traindata3,pWbin,3,petalwidthmin,intsz4)
result3,misc3=classify(testdata3,spLbin3,spWbin3,pLbin3,pWbin3,count3,75,sepallengthmin,sepalwidthmin,petallengthmin,petalwidthmin,intsz1,intsz2,intsz3,intsz4)

print "Misclassification percentage for Traindata3="
print float(misc3/75.0)*100
##########################################################################################

    
