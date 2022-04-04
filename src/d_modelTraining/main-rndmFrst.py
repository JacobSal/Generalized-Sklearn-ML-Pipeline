# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
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

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import   accuracy_score,f1_score
from sklearn.ensemble import  RandomForestClassifier

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from localPkg.datmgmt import DataManager

#%% PATHS 
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

def objective(space):
    clf=RandomForestClassifier(
                    n_estimators =space['n_estimators'], max_depth = space['max_depth'], min_samples_split = int(space['min_samples_split']))
                    #max_leaf_nodes = int(space['max_leaf_nodes']), min_samples_leaf= int(space['min_samples_leaf']),
                    #max_features = int(space['max_features']))
    
    
    clf.fit(X_train, y_train)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }
#enddef

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

#%% CREATE RANDOMFOREST PIPELINE
# print("starting modeling career...")
# coef = [671,10,68,3,650,87,462]
# RFmodel = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
#                                        max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
#                                        n_estimators = coef[4], max_samples = coef[5],
#                                        max_features = coef[6])

#%% GRID SEARCH
print("Gridsearch with cross-validation initializing...")

#Parameter Grid with ranges
paraGrid = {'max_depth': np.arange(10,210,20),
              'min_samples_split': np.arange(100,400,20),
              'max_leaf_nodes': np.arange(60,210,50),
              'min_samples_leaf': np.arange(10,50,10),
              'n_estimators': np.arange(100,200,50),
              'max_samples': np.arange(0.1,0.9,0.4)}
                #'max_features': np.arange(5,10,5)}


#Gridsearch with cross-validation
rf_gridsearch = GridSearchCV(estimator = RandomForestClassifier(), 
                        param_grid = paraGrid, 
                        scoring = 'f1', # can use make_scorer(custom_scoring_function, greater_is_better=True)
                        cv = 2, 
                        verbose = 2, 
                        n_jobs = -1)

#Calling Method 
# plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

rf_gridsearch.fit(X_train,y_train)
bestparams = rf_gridsearch.best_params_
print(bestparams)

#Plotting parameter performance

# grid_scores = ref_gridsearch.cv_results_
# plot.grid_search(rf_gridsearch.cv_results_, change=('max_depth', 'min_samples_split'),
#                  subset={'max_leaf_nodes': ''})

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
rf_gridsearch.fit(X_train,y_train)
bestparams = rf_gridsearch.best_params_
print(bestparams)

print('fitting...')
RFmodel = RandomForestClassifier(**bestparams)
model = RFmodel.fit(X_train,y_train)
#y_score = model.decision_function(X_test)
print(model.score(X_test,y_test))
filename = join(modelDir,('fittedRF_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done')


y_train_predict = model.predict(X_train)
y_predict = model.predict(X_test)
print('RF Train accuracy',accuracy_score(y_train, y_train_predict))
print('RF Test accuracy',accuracy_score(y_test,y_predict))
print('RF Train F1 Score', f1_score(y_train, y_train_predict))
print('RF Test F1 score', f1_score(y_test, y_predict))


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