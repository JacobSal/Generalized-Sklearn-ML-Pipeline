# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.20
"""
#%% IMPORTS
import time
import multiprocessing as mp
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import cv2
import dill as pickle

from os.path import dirname, join, abspath
from os import mkdir
from datetime import date
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve,auc
from sklearn.preprocessing import RobustScaler
from sklearn_evaluation import plot

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from localPkg.datmgmt import DataManager

#%% PATHS 
print("Number of processors: ", mp.cpu_count())
# Path to file
cfpath = dirname(__file__) 
# Path to images to be processed
folderName = abspath(join(cfpath,"..","a_dataGeneration","rawData"))
# Path to save bin : saves basic information
saveBin = join(cfpath,"saveBin")
# Path to training files
trainDatDir = abspath(join(cfpath,"..","b_dataAggregation","processedData","EL-11122021"))
# Path to model
modelDir = abspath(join(saveBin,"saveSVM"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
# Make directory for saves
try:
  mkdir(abspath(join(modelDir)))
except FileExistsError:
  print('Save folder for model already exists!')
#endtry

#%% DEFINITIONS & PARAMS
# def objective(space):
#     clf= someClassifier(
#                     # n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
#                     # reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
#                     # colsample_bytree=int(space['colsample_bytree']))
    
#     evaluation = [( X_train, y_train), ( X_test, y_test)]
    
#     clf.fit(X_train, y_train,
#             eval_set=evaluation, eval_metric="auc",
#             early_stopping_rounds=10,verbose=False)
    

#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, pred>0.5)
#     print ("SCORE:", accuracy)
#     return {'loss': -accuracy, 'status': STATUS_OK }
# #enddef

def plot_roc_curve(roc_auc_train, roc_auc_test):
    print('Generating ROC/AUC Plot...')
    plt.figure(0)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_tr, tpr_tr, 'g', label = 'Training AUC = %0.2f' % roc_auc_train)
    plt.plot(fpr_ts, tpr_ts, 'b', label = 'Testing AUC = %0.2f' % roc_auc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



#%% PARAMS
dTime = '19032022' #date.today().strftime('%d%m%Y')
#%% GRIDSEARCH PARAMS

# # param_range = [0.0001,0.001,0.01,0.1,1,10,100,1000]
# # param_range= np.arange(0.01,1,0.001)
# param_range2_C = [40,50,60,70,80,90,100,110,120,130,140]
# param_range2_ga = [0.0005,0.0006,0.0007,0.001,0.002,0.003,0.004]
# deg_range = [2,3,4,5,6,7]
# deg_range2 = [2,3,4,5,6,10]
# poly_range = np.arange(2,10,1)
# poly_range_C = np.arange(1e-15,1e-7,6e-10)
# poly_range_ga = np.arange(1e8,1e15,6e12)
# # param_grid = [{'svc__C':param_range,
# #                 'svc__kernel':['linear']},
# #               {'svc__C': param_range,
# #                 'svc__gamma':param_range,
# #                 'svc__kernel':['rbf']},
# #               {'svc__C': param_range,
# #                 'svc__gamma':param_range,
# #                 'svc__kernel':['poly'],
# #                 'svc__degree':deg_range}]
# param_grid2 = [{'svc__C': param_range2_C,
#                 'svc__gamma':param_range2_ga,
#                 'svc__decision_function_shape':['ovo','ovr'],
#                 'svc__kernel':['rbf']},
#                 {'svc__C': param_range2_C,
#                   'svc__gamma':param_range2_ga,
#                   'svc__kernel':['poly'],
#                   'svc__decision_function_shape':['ovo','ovr'],
#                   'svc__degree':deg_range2}]
# # param_grid2 = [{'svc__C': param_range2_C,
# #                 'svc__gamma':param_range2_ga,
# #                 'svc__decision_function_shape':['ovo','ovr'],
# #                 'svc__kernel':['rbf']}]
# # param_grid3 = [{'svc__C': poly_range_C,
# #                 'svc__gamma':poly_range_ga,
# #                 'svc__kernel':['poly'],
# #                 'svc__degree':poly_range}]

#%%
tmpSaveDir = join(cvDatDir, ('CVjoined_data_'+dTime+'.pkl'))
tmpSave = DataManager.load_obj(tmpSaveDir)
X_train = tmpSave[0]
X_test = tmpSave[1]
y_train = tmpSave[2]
y_train = y_train.reshape(len(y_train),1)
y_test = tmpSave[3]
y_test = y_test.reshape(len(y_test),1)
X = np.vstack((X_train,X_test))
y = np.ravel(np.vstack((y_train,y_test)))
print("y_train: " + str(np.unique(y_train)))
print("y_test: " + str(np.unique(y_test)))

#%% Create SVM Pipeline
# pipe_svc = make_pipeline(RobustScaler(),SVC(),verbose=True)
print('SVM:')

#%% SVM MODEL FITTING
# Create an instance of SVM and fit out data.
print("starting modeling career...")

#%% GRIDSEARCH
print("Gridsearch with cross-validation initializing...")

#Parameter Grid with ranges
paraGrid = {'C': [0.1, 1], #10, 100, 1000],
            # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf', 'sigmoid', 'linear']}

svm_gridsearch = GridSearchCV(estimator = SVC(),
                  param_grid = paraGrid,
                  scoring = 'roc_auc',
                  cv = 2,
                  n_jobs = 7,
                  verbose = 10)


svm_gridsearch.fit(X_train,y_train)
best_score = svm_gridsearch.best_score_
best_params = svm_gridsearch.best_params_
print('Best Params: ', best_params)
print('Best Score: ', best_score)
print('CV Results: ', svm_gridsearch.cv_results_)

#Plotting parameter performance
print('Plotting Gridsearch Results...')
grid_scores = svm_gridsearch.cv_results_

#%% PARAMETER SETTING (IF AVAILABLE)
### Setting Parameters ###
# print('fitting...')
# #{'svc__C': 100, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'} (~0.72% f1_score)
# #{svc__C=130, svc__decision_function_shape=ovr, svc__gamma=0.0005, svc__kernel=rbf}

# pipe_svc.set_params(svc__C =  130, 
#                     svc__gamma = 0.0005, 
#                     svc__kernel =  'rbf',
#                     svc__probability = True,
#                     svc__shrinking = False,
#                     svc__decision_function_shape = 'ovr')

#%% MODEL FITTING
print('fitting...')
clf = SVC(**best_params)
clf.fit(X_train,y_train)
#y_score = model.decision_function(X_test)
print(clf.score(X_test,y_test))
filename = join(modelDir,('fittedSVM_'+dTime+'.sav'))
pickle.dump(clf, open(filename, 'wb'))
print('done')

#Result metrics 
print('Generating Scores...')
y_train_predict = clf.predict(X_train)
y_predict = clf.predict(X_test)
print('SVM Train accuracy',accuracy_score(y_train, y_train_predict))
print('SVM Test accuracy',accuracy_score(y_test,y_predict))
print('SVM Train F1 Score', f1_score(y_train, y_train_predict))
print('SVM Test F1 score', f1_score(y_test, y_predict))
print('SVM Train ROC_AUC Score', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
print('SVM Test ROC_AUC score', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

#train data ROC
fpr_tr, tpr_tr, threshold = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
roc_auc_train = auc(fpr_tr, tpr_tr)

#test data ROC
fpr_ts, tpr_ts, threshold = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc_test = auc(fpr_ts, tpr_ts)

#Plot ROC curve
plot_roc_curve(roc_auc_train, roc_auc_test)
#%% CROSS VALIDATION
# scores = cross_val_score(estimator = model,
#                           X = X,
#                           y = y,
#                           cv = 10,
#                           scoring = 'roc_auc',
#                           verbose = 5,
#                           n_jobs=-1)

# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
