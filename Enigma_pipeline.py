#!/usr/bin/env python
# coding: utf-8

# Import libraries and create constants

from enigma_data import enigma_data
from Model_Factory import Model_Factory
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
import Constants as constants
import numpy as np
import argparse
import os
import pandas as pd
from joblib import dump, load
from datetime import datetime

NB_DEBUG = 1

# ---------------------------------------------------------------------------
# Computing environment (CE) stuff
# ---------------------------------------------------------------------------
# Parse arguments and create save files

parser = argparse.ArgumentParser()
parser.add_argument("-sdx", type=int, choices=range(3,7), default=3, help="Sdx number (3,4,5,6)")
parser.add_argument("-gpu", type=int, choices=range(0,4), default=0, help="Assign to gpu device (0,1,2, or 3)")
parser.add_argument("-csv", default="all.csv", help="Enigma CSV file")
args = parser.parse_args()

sdx = args.sdx
gpu_dev = str( args.gpu )
csv_file = args.csv

if NB_DEBUG:
	print( "SDx={0:d}, GPU={1:s}, CSV File={2:s}".format( sdx, gpu_dev, csv_file ) )

if not os.path.exists( './archive' ):
    if NB_DEBUG:
        print( "joblib archive folder does not exist, making one [./archive]" )
    os.makedirs( './archive' )

ts = np.int32( datetime.timestamp( datetime.now() ) )

gs_file = "./archive/grids_result_" + str( ts ) + ".joblib"
cv_file = "./archive/cvf_result_" + str( ts ) + ".joblib"

if NB_DEBUG:
    print( "GridSearch File = {0:s}, CVFold File = {1:s}".format( gs_file, cv_file ) )

if NB_DEBUG: 
    print( "GPU Device = {0:s}".format( gpu_dev ) )
    
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_dev

# ---------------------------------------------------------------------------
# Construct pipeline, identify optimal model parameters, evaluate accuracy, 
# and identify input features that contribute to model accuracy
# ---------------------------------------------------------------------------

# [Step 1] Parse csv file and create engima data 


if NB_DEBUG: 
    print( "Sdx label = {0:d}".format( sdx ) )

data = enigma_data( dfile=csv_file, predict='SDx', harmonize=True, predict_val=sdx, 
                    feature_range=[ 'ICV', 'R_insula_surfavg' ],
                     batch='Site', covariates=['Sex','Age'] )

X_train, X_test, Y_train, Y_test = data.parse( remove_columns=["SubjID","Dx","Handedness","DURILL", "AO"] ).partition()
features = X_train.columns


# [Step 2] Create the models and pipeline


kfold = StratifiedKFold( n_splits=constants.CV_FOLDS, shuffle=True )

factory = Model_Factory()

dl_model = factory.build( model_type=constants.MODELS.DL_CLASSIFY, input_dim=X_train.shape[1]  )
imbalance_model = factory.build( model_type=constants.MODELS.IMBALANCE )
scaler_model = factory.build( model_type=constants.MODELS.SCALER )

keras_model = KerasClassifier( build_fn=dl_model.construct, 
                               learn_rate=0.15,
                               epochs=100,
                               hidden_units_L1=50,
                               hidden_units_L2=2,
                               l2_reg_penalty=0.8,
                               drop_out_rate = 0.5,
                               validation_split=0.25,
                               batch_size = 20,
                               verbose=1 )

pipeline = Pipeline( [ ('scaler', scaler_model ), ('imbalance', imbalance_model ),
                       ('classifier', keras_model ) ], memory=None )


# [Step 3] Perform grid search

grid = GridSearchCV( estimator=pipeline, param_grid=dl_model.get_grid_dict(), cv=kfold, verbose=2 )
grid_result = grid.fit( X_train, Y_train )
dump( grid_result, gs_file ) # save to file -- so we can drill down into the results a bit more 

print( grid_result.best_params_ ) # quickly display the optimal hyper-parameter combination


# [Step 4] Verify grid search performance using k-fold cross-validation and optimal model parameters

grid_result = load( gs_file )

keras_model = KerasClassifier( build_fn=dl_model.construct, 
                               learn_rate=grid_result.best_params_['classifier__learn_rate'],
                               epochs=grid_result.best_params_['classifier__epochs'],
                               hidden_units_L1=grid_result.best_params_['classifier__hidden_units_L1'],
                               hidden_units_L2=grid_result.best_params_['classifier__hidden_units_L2'],
                               l2_reg_penalty=grid_result.best_params_['classifier__l2_reg_penalty'],
                               drop_out_rate=grid_result.best_params_['classifier__drop_out_rate'],
                               batch_size=20,
                               validation_split=0.25,
                               verbose=1 )

pipeline = Pipeline( [ ('scaler', scaler_model ), ('imbalance', imbalance_model ),
                       ('classifier', keras_model ) ], memory=None )

results = cross_validate( pipeline, X_train, Y_train, cv=kfold, 
                          scoring=('accuracy'), return_estimator=True, verbose=1 )

dump( results, cv_file )     

print( "train classification accuracy = {0:.2f} +/- {1:.2f}".format( np.mean( results['test_score'] ), np.std( results['test_score'] ) ) )


# [Step 5] Calculate and print the model performance metrics


PERF=np.zeros( ( 2, constants.CV_FOLDS ) )
for idx in range( 0, constants.CV_FOLDS ):
    PERF[0,idx]=accuracy_score( Y_test, results['estimator'][idx]['classifier'].predict( X_test ) )
    PERF[1,idx]=roc_auc_score( Y_test, results['estimator'][idx]['classifier'].predict( X_test ) )
    
print( PERF )
print( "Test classification accuracy = {0:.2f} +/- {1:.2f}".format( np.mean( PERF[0,:] ), np.std( PERF[0,:] ) ) )
print( "Test classification roc = {0:.2f} +/- {1:.2f}".format( np.mean( PERF[1,:] ), np.std( PERF[1,:] ) ) )


# [Step 6] Select input features that contribute to classification accuracy using DNN backtrack technique

results = load( cv_file ) 
dl_model.construct( hidden_units_L1=grid_result.best_params_['classifier__hidden_units_L1'], hidden_units_L2=grid_result.best_params_['classifier__hidden_units_L2'] )
idx, w = dl_model.select_features( results, features, top_num_features=30, weight_matrix_threshold=0.01 )


# Bye ... bye




