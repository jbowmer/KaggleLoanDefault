# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:16:07 2015

@author: Jake
"""

#A script to generate predictions for the Kaggle Loan Default Prediction competition.
#https://www.kaggle.com/c/loan-default-prediction/

#The model is two staged - first, I train a classifier to predict default.
#The second model predicts loss given default.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#import classifiers and refgressors.
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score


#import data split
from sklearn.cross_validation import train_test_split


os.chdir('/Users/Jake/Kaggle/LoanDefault/Data')
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
os.chdir('/Users/Jake/Kaggle/LoanDefault')

train.head()
test.head()

#Create two new columns:
train['first'] = train.f528 - train.f274
train['second'] = train.f528 - train.f527
test['first'] = test.f528 - test.f274
test['second'] = test.f528 - test.f527

#create a new column - 'default' which is 1 if loss is > 0
train['default'] = train.loss.map(lambda x: 1 if x > 0 else 0)

#Split the training set into train and validation:
train_features = train[['f2', 'f22', 'f274', 'f271', 'f527', 'f528', 'first', 'second']]
train_target = train['default'].astype('float')

#Clean train_features:

for x in list(train_features):
    if train_features[x].dtype not in ['float64']:
        train_features[x] = train_features[x].astype(float)
    if not all(train_features[x].notnull()):
        train_features[x] = train_features[x].fillna(train_features[x].median())
   
#Split data into train and validation sets:
choosen_random_state = 1
X_train, X_validate, Y_train, Y_validate = train_test_split(train_features,train_target, 
                                                            test_size = 0.3, 
                                                            random_state=choosen_random_state)

#Train the classifiers:
rf_mod_gini = RandomForestClassifier(n_estimators = 100, n_jobs = -1, criterion = 'gini')
rf_mod_ent = RandomForestClassifier(n_estimators = 100, n_jobs = -1, criterion = 'entropy')
et_mod_gini = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
et_mod_ent = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy')
gbm_mod = GradientBoostingClassifier(subsample=0.5, max_depth=6, n_estimators=50)

#Cross validation accuracy of all models:

cv_folds = 5
eval_scoring = 'accuracy'
workers = -1


rf_mod_gini_scores = cross_val_score(rf_mod_gini, X_train, Y_train, cv=cv_folds, 
                                     scoring = eval_scoring, n_jobs=workers)
print (np.mean(rf_mod_gini_scores), np.std(rf_mod_gini_scores))

rf_mod_ent_scores = cross_val_score(rf_mod_ent, X_train, Y_train, cv=cv_folds, 
                                     scoring = eval_scoring, n_jobs=workers)
print (np.mean(rf_mod_ent_scores), np.std(rf_mod_ent_scores))
et_mod_gini_scores = cross_val_score(et_mod_gini, X_train, Y_train, cv=cv_folds, 
                                     scoring = eval_scoring, n_jobs=workers)                                     
print (np.mean(et_mod_gini_scores), np.std(et_mod_gini_scores))
et_mod_ent_scores = cross_val_score(et_mod_ent, X_train, Y_train, cv=cv_folds, 
                                     scoring = eval_scoring, n_jobs=workers)
print (np.mean(et_mod_ent_scores), np.std(et_mod_ent_scores))
gbm_mod_scores = cross_val_score(gbm_mod, X_train, Y_train, cv=cv_folds, 
                                     scoring = eval_scoring, n_jobs=workers)
print (np.mean(gbm_mod_scores), np.std(gbm_mod_scores))

#Accuracy scores are 98-98.5% for all models in sample. Use holdout data to check:
#Fit all models:
rf_mod_gini.fit(X_train, Y_train)
rf_mod_ent.fit(X_train, Y_train)
et_mod_gini.fit(X_train, Y_train)
et_mod_ent.fit(X_train, Y_train)
gbm_mod.fit(X_train, Y_train)

#
Y_pred_rf_mod_gini = rf_mod_gini.predict(X_validate)
Y_pred_rf_mod_ent = rf_mod_ent.predict(X_validate)
Y_pred_et_mod_gini = et_mod_gini.predict(X_validate)
Y_pred_et_mod_ent = et_mod_ent.predict(X_validate)
Y_pred_gbm_mod = gbm_mod.predict(X_validate)

#accuracy:
rf_gini_accuracy =  accuracy_score(Y_validate, Y_pred_rf_mod_gini)
rf_ent_accuracy = accuracy_score(Y_validate, Y_pred_rf_mod_ent)
et_gini_accuracy = accuracy_score(Y_validate, Y_pred_et_mod_gini)
et_ent_accuracy = accuracy_score(Y_validate, Y_pred_et_mod_ent)
gbm_accuracy = accuracy_score(Y_validate, Y_pred_gbm_mod)



#Now get a list of probabilities:

Y_pred_rf_mod_gini = rf_mod_gini.predict_proba(X_validate)[:,1]
Y_pred_rf_mod_ent = rf_mod_ent.predict_proba(X_validate)[:,1]
Y_pred_et_mod_gini = et_mod_gini.predict_proba(X_validate)[:,1]
Y_pred_et_mod_ent = et_mod_ent.predict_proba(X_validate)[:,1]
Y_pred_gbm_mod = gbm_mod.predict_proba(X_validate)[:,1]

#Create dataframe then average them:
predictions = pd.DataFrame({'rf1': Y_pred_rf_mod_gini, 'rf2' : Y_pred_rf_mod_ent, 'et1' : Y_pred_et_mod_gini,
                            'et2' : Y_pred_et_mod_ent, 'gbm' : Y_pred_gbm_mod}) 

predictions['average'] = predictions.mean(axis = 1)

#WE GO THROUGH THE SAME PROCESS WHEN DETERMINING LOSS.
#SUBMISSIONS ARE MADE ON THE BASIS THAT IF DEFAULT PROB IS OVER 0.5, WE SUBMIT THE LOSS REGRESSION.
#OTHERWISE WE SUBMIT 0.
