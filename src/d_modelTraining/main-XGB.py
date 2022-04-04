# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from os import mkdir
import dill as pickle

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import   accuracy_score
from sklearn.ensemble import  RandomForestClassifier

from xgboost import XGBClassifier

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
modelDir = abspath(join(saveBin,"saveDT"))
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
    clf=XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }
#enddef

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

#%% XGBOOST ALGORITHM 
print('XGBoost:')

##XGBoost with Optimal HyperParameters
print("starting modeling career...")
coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]
XGBmodel = XGBClassifier(max_depth = coef[0],subsample = coef[1],n_estimators = coef[2],
                      colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate = coef[5], 
                      min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
                      reg_lambda = coef[9])

#%% GRID SEARCH
# print("Gridsearch with cross-validation initializing...")

# #Parameter Grid with ranges
# param_grid = {'max_depth': [2,50], #(0,9999), typical: 3-10  
#               'subsample': [0.5,1], #[0,1], typical values: 0.5-1  
#               'n_estimators': [50,200], #[0,9999], 
#               'colsample_bylevel': [0.5,1], #[0,1] ,
#               'colsample_bytree': [0.5,1], #[0,1],
#                'learning_rate': [0.1,0.36] , #(0,1), typical: 0.01-0.2
#                'min_child_weight': [1,50], #(0,9999),
#                'alpha': [0.5,1], #[0,1],
#                'lambda': [0.5,1], #[0,1]}

# #Gridsearch with cross-validation
# xgb_gridsearch = GridSearchCV(estimator = XGBClassifier(), 
#                         param_grid = param_grid,
#                         scoring = 'f1',
#                         cv = 1, 
#                         verbose = 2, 
#                         n_jobs = -1)

# #xgb_grid.fit(X_train,y_train)
# bestparams = xgb_gridsearch.best_params_
# print(bestparams)

#%% BAYESIAN OPTIMIZATION WITH HYPEROPT

#Domain space
space = {'max_depth': hp.quniform("max_depth",3,18,1), #(0,9999), typical: 3-10
              'gamma': hp.uniform ('gamma', 1,9),
              'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
               'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
               'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
               'reg_lambda' : hp.uniform('reg_lambda', 0,1),
               'n_estimators': 180,
               'seed': 0
               }

#Run optimization algorithm
print("Bayesian optimization initializing...")
trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)




#%% MODEL FITTING
print('fitting...')
model = XGBmodel.fit(X_train,y_train)
#y_score = model.decision_function(X_test)
print(model.score(X_test,y_test))
filename = join(modelDir,('fittedXGB_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done')

y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
print('XGB Train accuracy',accuracy_score(y_train, y_train_predict))
print('XGB Test accuracy',accuracy_score(y_test,y_predict))

#%% CROSS VALIDATE k-fold (k=10)
# scores = cross_val_score(estimator = model,
#                           X = X_train,
#                           y = y_train,
#                           cv = 10,
#                           scoring = 'roc_auc',
#                           verbose = True,
#                           n_jobs=-1)

# print('XGB CV accuracy scores: %s' % scores)
# print('XGB CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 
        
#Best coefficients so far:
    #coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]

"""
#%% SAMPLE CODE FOR OPTIMIZING PARAMETERS
score = 0.7159090909090909       
coef = [2,0.4,150,.8,1,.1,1,0,1,0.5]
for ii in range(2,31): 
    model = XGBClassifier(max_depth = ii,subsample = coef[1],n_estimators = coef[2],
                        colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate=coef[5], 
                        min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
                        reg_lambda = coef[9])
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