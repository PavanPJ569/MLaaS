import pandas as pd
import numpy as np

import sklearn
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer,OneHotEncoder,LabelEncoder
from sklearn.cluster import Birch
import pickle as cPickle

def zero():
    LR=DecisionTreeClassifier()
    return LR
 
def one():
    LR=ExtraTreeClassifier()
    return LR
 
def two():
    LR=KNeighborsClassifier()
    return LR

def three():
    LR=LinearSVC()
    return LR

def four():
    LR=MLPClassifier()
    return LR

def five():
    LR=GradientBoostingClassifier()
    return LR

def six():
    LR = RandomForestClassifier()
    return LR    

def seven():
    cart=DecisionTreeClassifier()
    LR= BaggingClassifier(base_estimator=cart)
    return LR

def eight():
    LR=AdaBoostClassifier()
    return LR

def training(classifier):
    train=pd.read_csv('dataset/train.csv', sep=',')
    test=pd.read_csv('dataset/test.csv', sep=',')
    x_train=train.iloc[:, :-1]
    y_train=train.iloc[:,-1]
    x_test=test.iloc[:,1:-1]
    for column in x_train.columns:
        if x_train[column].dtype == type(object):
            le = LabelEncoder()
            x_train[column] = le.fit_transform(x_train[column])
    for column in x_test.columns:
        if x_test[column].dtype == type(object):
            le = LabelEncoder()
            x_test[column] = le.fit_transform(x_test[column])
    switcher={
        'decisiontree' : zero,
        'extratree' : one,
        'knn' : two,
        'linearsvc' : three,
        'mlp' : four,
        'gradientboosting' : five,
        'randomforest' : six,
        'bagging' : seven,
        'adaboost' : eight
    }
    select=switcher[classifier]
    LR=select()
    score2=LR.fit(x_train, y_train)
    return LR.predict(x_test)


