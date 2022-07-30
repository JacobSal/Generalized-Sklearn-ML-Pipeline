# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm/eduluca
"""

import numpy as np
import multiprocessing as mp
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from os import mkdir
import dill as pickle
from ttictoc import tic,toc

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve,auc,plot_confusion_matrix,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn_evaluation import plot

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from localPkg.datmgmt import DataManager

tic()
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
modelDir = abspath(join(saveBin,"saveRF"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
# Make directory for saves
try:
    mkdir(abspath(join(modelDir)))
except FileExistsError:
  print('Save folder for model already exists!')
#endtry

#%% DEFINITIONS & PARAMS
def robust_save(fname):
    plt.savefig(join(saveBin,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
#enddef

# def objective(space):
#     clf=RandomForestClassifier(
#                     n_estimators =space['n_estimators'], max_depth = space['max_depth'], min_samples_split = int(space['min_samples_split']))
#                     #max_leaf_nodes = int(space['max_leaf_nodes']), min_samples_leaf= int(space['min_samples_leaf']),
#                     #max_features = int(space['max_features']))
    
    
#     clf.fit(X_train, y_train)
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

# def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
#     # Get Test Scores Mean and std for each grid search
#     scores_mean = cv_results['mean_test_score']
#     scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

#     scores_sd = cv_results['std_test_score']
#     scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

#     # Plot Grid search scores
#     _, ax = plt.subplots(1,1)

#     # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
#     for idx, val in enumerate(grid_param_2):
#         ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

#     ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
#     ax.set_xlabel(name_param_1, fontsize=16)
#     ax.set_ylabel('CV Average Score', fontsize=16)
#     ax.legend(loc="best", fontsize=15)
#     ax.grid('on')
# #enddef

#%% PARAMS
dTime = '19032022' #date.today().strftime('%d%m%Y')

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


#%% RANDOM FOREST ALGORITHM 
print('Random Forest:')

#%% CREATE RANDOMFOREST PIPELINE (first set vs second set of data)
# print("starting modeling career...")
# coef = [671,10,68,3,650,87,462]
# clf = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
#                                        max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
#                                        n_estimators = coef[4], max_samples = coef[5],
#                                        max_features = coef[6])


print("starting modeling career...")
clf = RandomForestClassifier(criterion = 'gini', max_features = 'sqrt', max_samples = 0.5, 
                              min_samples_leaf = 1, min_samples_split = 2, n_estimators = 900)


#%% GRID SEARCH
# print("Gridsearch with cross-validation initializing...")

# #Parameter Grid with ranges
# paraGrid = {'criterion': ['gini','entropy'],
#             'min_samples_leaf': np.arange(1,15,1),
#             'min_samples_split': np.arange(2,20,2),
#             'n_estimators': np.arange(300,901,200),
#             'max_features': ['auto','sqrt','log2'],
#             'max_samples': np.arange(0.1,0.9,0.4)} #18144 fits
              
#             # 'max_depth': np.arange(10,210,20),
#             # 'max_leaf_nodes': np.arange(10,210,50),

              


# #Gridsearch with cross-validation
# rf_gridsearch = GridSearchCV(estimator = RandomForestClassifier(), 
#                         param_grid = paraGrid, 
#                         scoring = 'roc_auc', # can use make_scorer(custom_scoring_function, greater_is_better=True)
#                         cv = 3, 
#                         verbose = 10, 
#                         n_jobs = 30)

# #Calling Method 
# # plot_grid_search(rf_gridsearch.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

# rf_gridsearch.fit(X_train,y_train)
# best_score = rf_gridsearch.best_score_
# best_params = rf_gridsearch.best_params_
# print('Best Params: ', best_params)
# print('Best Score: ', best_score)
# print('CV Results: ', rf_gridsearch.cv_results_)

# #Plotting parameter performance
# print('Plotting Gridsearch Results...')
# grid_scores = rf_gridsearch.cv_results_

# # plt.figure(0)
# # plot.grid_search(grid_scores, change=('min_samples_leaf', 'min_samples_split'),
# #                  subset={'n_estimators': [100]})

# # plt.figure(1)
# # plot.grid_search(grid_scores, change='n_estimators', kind='bar')


#%% BAYESIAN OPTIMIZATION WITH HYPEROPT

# #Domain space
# space = {'max_depth': hp.quniform("max_depth",10,50,10), #(0,9999), typical: 3-10
#               'min_samples_split': hp.uniform ('min_samples_split',8,10),
#               'max_leaf_nodes' : hp.quniform('max_leaf_nodes', 50,100,10),
#                'min_samples_leaf' : hp.quniform('min_samples_leaf', 1, 6, 1),
#                'max_features' : hp.quniform('max_features', 50,800,50),
#                'n_estimators': 200,
#                'seed': 0
#                }

# #Run optimization algorithm
# print("Bayesian optimization initializing...")
# trials = Trials()


# best_hyperparams = fmin(fn = objective,
#                         space = space,
#                         algo = tpe.suggest,
#                         max_evals = 100,
#                         trials = trials)

# print("The best hyperparameters are : ","\n")
# print(best_hyperparams)



#%% MODEL FITTING
print('fitting...')
# clf = RandomForestClassifier(**best_params)
clf.fit(X_train,y_train)
#y_score = model.decision_function(X_test)
print(clf.score(X_test,y_test))
filename = join(modelDir,('fittedRF_'+dTime+'.sav'))
pickle.dump(clf, open(filename, 'wb'))
print('done')

#Result metrics 
print('Generating Scores...')
y_train_predict = clf.predict(X_train)
y_predict = clf.predict(X_test)
print('RF Train accuracy',accuracy_score(y_train, y_train_predict))
print('RF Test accuracy',accuracy_score(y_test,y_predict))
print('RF Train F1 Score', f1_score(y_train, y_train_predict))
print('RF Test F1 score', f1_score(y_test, y_predict))
print('RF Train ROC_AUC Score', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
print('RF Test ROC_AUC score', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

#train data ROC
fpr_tr, tpr_tr, threshold = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
roc_auc_train = auc(fpr_tr, tpr_tr)

#test data ROC
fpr_ts, tpr_ts, threshold = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc_test = auc(fpr_ts, tpr_ts)

#Plot confusion matrix and scores
tn, fp, fn, tp = confusion_matrix(list(y_test), list(y_predict), labels=[0, 1]).ravel()
# print('True Positive', tp)
# print('True Negative', tn)
# print('False Positive', fp)
# print('False Negative', fn)
tot = tn+tp+fp+fn
print('True Positive Rate', tp/tot)
print('True Negative Rate', tn/tot)
print('False Positive Rate', fp/tot)
print('False Negative Rate', fn/tot)
# plot_confusion_matrix(clf, y_test, y_predict)

#Plot ROC curve
plot_roc_curve(roc_auc_train, roc_auc_test)

print('Time to run:',toc())

#%% Cross Validate
# scores = cross_val_score(estimator = model,
#                           X = X,
#                           y = y,
#                           cv = 10,
#                           scoring = 'roc_auc',
#                           verbose = True,
#                           n_jobs=-1)

# print('RF CV accuracy scores: %s' % scores)
# print('RF CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

#Best coefficients so far:
    #coef = [671,10,68,3,650,87,462]


#%% SAMPLE CODE FOR OPTIMIZING PARAMETERS

"""
score = 0.75       
coef = [671,10,68,3,650,192,462]
for ii in range(2,500,10): 
        model = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
                                        max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
                                        n_estimators = coef[4], max_samples = coef[5],
                                        max_features = ii)
        model.fit(X_train,y_train) 
        y_predict = model.predict(X_test) 
        y_train_predict = model.predict(X_train)
        newscore = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        print(ii,newscore, end="")
        if newscore > score:
            print(' best so far')
            score = newscore
        else:
            print()
"""
