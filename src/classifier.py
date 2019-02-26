# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#from sklearn.svm import LinearSVC
from sklearn.svm import SVC



def get_classifier(method='logistic_regression'):
    if 'logistic_regression' == method:
        return LogisticRegression(C=1e3,
                                  tol=0.01,
                                  multi_class='ovr',
                                  solver='liblinear',
                                  n_jobs=-1,
                                  random_state=123)
    if 'random_forest' == method:
        return RandomForestClassifier(n_estimators=250,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=123)

    if 'svm' == method:
#        return LinearSVC(tol=1e-4,  
#                         C = 0.10000000000000001, 
#                         penalty='l2', 
#                         class_weight={1:1.0, 2:0.9})
        return SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        
def cross_validation(model, data, class_label, cv):
    return cross_val_score(model, data, class_label, cv)


