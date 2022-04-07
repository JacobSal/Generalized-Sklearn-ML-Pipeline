# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:19:38 2021

@author: jsalm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""
from os import mkdir
from os.path import join, abspath, dirname

import numpy as np
import matplotlib.pyplot as plt
import cv2
import dill as pickle


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn_evaluation import plot

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from localPkg.preproc import ProcessPipe
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
modelDir = abspath(join(saveBin,"saveKNN"))
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
#     clf=KNeighborsClassifier(
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

#%% Create KNN pipeline
print('KNN:')
# pipe_knn = make_pipeline(RobustScaler(),KNeighborsClassifier())

#%% KNN MODEL FITTING
#we create an instance of KNN and fit out data.
print("starting modeling career...")

#%% GRIDSEARCH
print("Gridsearch with cross-validation initializing...")

#Parameter Grid with ranges
paraGrid = {'n_neighbors': np.arange(1,5,2),
            'weights': ['uniform', 'distance']}
            # 'metric': ['euclidean', 'manhattan']}

knn_gridsearch = GridSearchCV(estimator = KNeighborsClassifier(),
                  param_grid = paraGrid,
                  scoring = 'roc_auc',
                  cv = 2,
                  n_jobs = -1,
                  verbose = 10)


knn_gridsearch.fit(X_train,y_train)
best_score = knn_gridsearch.best_score_
best_params = knn_gridsearch.best_params_
print('Best Params: ', best_params)
print('Best Score: ', best_score)
print('CV Results: ', knn_gridsearch.cv_results_)

#Plotting parameter performance
print('Plotting Gridsearch Results...')
grid_scores = knn_gridsearch.cv_results_


#%% MODEL FITTING
print('fitting...')
clf = KNeighborsClassifier(**best_params)
clf.fit(X_train,y_train)
#y_score = model.decision_function(X_test)
print(clf.score(X_test,y_test))
filename = join(modelDir,('fittedKNN_'+dTime+'.sav'))
pickle.dump(clf, open(filename, 'wb'))
print('done')

#Result metrics 
y_train_predict = clf.predict(X_train)
y_predict = clf.predict(X_test)
print('KNN Train accuracy',accuracy_score(y_train, y_train_predict))
print('KNN Test accuracy',accuracy_score(y_test,y_predict))
print('KNN Train F1 Score', f1_score(y_train, y_train_predict))
print('KNN Test F1 score', f1_score(y_test, y_predict))
print('KNN Train ROC_AUC Score', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
print('KNN Test ROC_AUC score', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

#train data ROC
fpr_tr, tpr_tr, threshold = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
roc_auc_train = auc(fpr_tr, tpr_tr)

#test data ROC
fpr_ts, tpr_ts, threshold = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc_test = auc(fpr_ts, tpr_ts)

#Plot ROC curve
plot_roc_curve(roc_auc_train, roc_auc_test)
#%% CROSS VALIDATE
# scores = cross_val_score(estimator = pipe_knn,
#                           X = X,
#                           y = y,
#                           cv = 10,
#                           scoring = 'roc_auc',
#                           verbose = True,
#                           n_jobs=-1)

# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 