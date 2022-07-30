# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm/eduluca
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from os import mkdir
import dill as pickle
from ttictoc import tic,toc

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,auc,plot_confusion_matrix,confusion_matrix
from sklearn.ensemble import  RandomForestClassifier
from sklearn_evaluation import plot

from xgboost import XGBClassifier

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

# def objective(space):
#     clf=XGBClassifier(
#                     n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
#                     reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
#                     colsample_bytree=int(space['colsample_bytree']))
    
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

# ##XGBoost with Optimal HyperParameters
# print("starting modeling career...")
# # coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]
# # clf = XGBClassifier(max_depth = coef[0],subsample = coef[1],n_estimators = coef[2],
# #                       colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate = coef[5], 
# #                       min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
# #                       reg_lambda = coef[9])



# clf = XGBClassifier(max_depth = 5, learning_rate = 0.1, n_estimators = 200,
#                     gamma = 0.01, subsample = 0.5, colsample_bylevel = 0.1, colsample_bytree = 0.7)
#%% GRID SEARCH
print("Gridsearch with cross-validation initializing...")

#Parameter Grid with ranges
paraGrid = {'max_depth': np.arange(1,15,2), #(0,9999), typical: 3-10  #15
            'learning_rate': np.arange(0.1,0.2,0.05),#(0,1), typical: 0.01-0.2 #0.5
            'n_estimators': np.arange(100,250,50), #[0,9999], 
            'gamma': np.arange(0.01,0.05,0.02),
            'subsample': np.arange(0.1,0.9,0.4),
            'colsample_bylevel': np.arange(0.1,0.9,0.4), #[0,1] ,
            'colsample_bytree': np.arange(0.1,0.9,0.2)} #[0,1],
            # 'alpha': np.arange(0.25,1,0.25), #[0,1],
            # 'lambda': np.arange(0.25,1,0.25)}#[0,1]}

#Gridsearch with cross-validation
xgb_gridsearch = GridSearchCV(estimator = XGBClassifier(), 
                        param_grid = paraGrid,
                        scoring = 'precision',
                        cv = 3, 
                        verbose = 8, 
                        n_jobs = 30)

#Calling Method 
# plot_grid_search(rf_gridsearch.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

xgb_gridsearch.fit(X_train,y_train)
best_score = xgb_gridsearch.best_score_
best_params = xgb_gridsearch.best_params_
print('Best Params: ', best_params)
print('Best Score: ', best_score)
print('CV Results: ', xgb_gridsearch.cv_results_)

#Plotting parameter performance
print('Plotting Gridsearch Results...')
grid_scores = xgb_gridsearch.cv_results_

# plt.figure(0)
# plot.grid_search(grid_scores, change=('min_samples_leaf', 'min_samples_split'),
#                  subset={'n_estimators': [100]})

# plt.figure(1)
# plot.grid_search(grid_scores, change='n_estimators', kind='bar')

#%% BAYESIAN OPTIMIZATION WITH HYPEROPT

# #Domain space
# space = {'max_depth': hp.quniform("max_depth",3,18,1), #(0,9999), typical: 3-10
#               'gamma': hp.uniform ('gamma', 1,9),
#               'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#                'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
#                'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
#                'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#                'n_estimators': 180,
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
clf = XGBClassifier(**best_params)
clf.fit(X_train,y_train)
#y_score = model.decision_function(X_test)
print(clf.score(X_test,y_test))
filename = join(modelDir,('fittedXGB_'+dTime+'.sav'))
pickle.dump(clf, open(filename, 'wb'))
print('done')

#Result metrics 
print('Generating Scores...')
y_train_predict = clf.predict(X_train)
y_predict = clf.predict(X_test)
print('XGB Train accuracy',accuracy_score(y_train, y_train_predict))
print('XGB Test accuracy',accuracy_score(y_test,y_predict))
print('XGB Train precision',precision_score(y_train, y_train_predict))
print('XGB Test precision',precision_score(y_test,y_predict))
print('XGB Train recall',recall_score(y_train, y_train_predict))
print('XGB Test recall',recall_score(y_test,y_predict))
print('XGB Train F1 Score', f1_score(y_train, y_train_predict))
print('XGB Test F1 score', f1_score(y_test, y_predict))
print('XGB Train ROC_AUC Score', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
print('XGB Test ROC_AUC score', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
print('Best Params: ', best_params)
print('Best Score: ', best_score)
print('CV Results: ', xgb_gridsearch.cv_results_)

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
