# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:55:56 2018

@author: Jaspreet
"""

import scipy.io as sio
import numpy as np

mat_contents_train = sio.loadmat('train.mat')
X=mat_contents_train['X_train_bag']
Y=mat_contents_train['Y_train']

Xtrain=X[0:15000,:]
Ytrain1=Y[0:15000]
Ytrain2=Ytrain1.T
Ytrain=Ytrain2[0,0:15000]
print(Ytrain.shape)


Xtest=X[15000:18001,:]
Ytest1=Y[15000:18001]
Ytest2=Ytest1.T
Ytest=Ytest2[0,0:15000]

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(Xtrain)
print(X_train_tfidf.shape)

# Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
text_clf = MultinomialNB().fit(X_train_tfidf, Ytrain)
predicted = text_clf.predict(Xtest)
print(np.mean(predicted == Ytest))

# Logistic regression classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train_tfidf, Ytrain)
predicted = clf.predict(Xtest)
print(np.mean(predicted == Ytest))
