# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:49:36 2020

@author: Srujan
"""
 
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re



data=pd.read_csv("C:/Users/Srujan/Documents/Datasets/smsspamcollection/SMSSpamCollection",sep='\t',names=['label','message'])

corpus=[]
ps=PorterStemmer()
for i in range(len(data)):
    review= data['message'][i]
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)

y=pd.get_dummies(data['label'])

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print(acc)