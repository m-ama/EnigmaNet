import os
import logging
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.model_selection import train_test_split

class enigma_data(object):

	def __init__( self, dfile, predict, predict_val, feature_range, harmonize=True, batch=None, covariates=None ):

		self.DEBUG = True

		if os.path.exists( dfile ):

			self.dframe = pd.read_csv( dfile )
      
			if self.DEBUG:
				print( 'Found classes: ' + str(self.dframe.loc[:,predict].unique()))
			else:
				assert isinstance( dfile, object )

		self.predict = predict		# what are we predicting
		self.predict_val = predict_val	# what is the prediction value (or label)
		self.feature_range = feature_range	# feature range (start and stop) in the spreadsheet
		self.harmonize = harmonize	# run combate (yes or no)

		if ( self.harmonize ) and ( batch is not None ) and ( covariates is not None ):
			self.covariates = covariates
			self.batch = batch
		elif self.harmonize:
			logging.warning("Batch or covariates is not defined ... unable to harmonize" )
			self.harmonize = False

	# --------------------------------------------------
	# These two methods are primarly used for debugging
	#	getFrame()
	# 	XY()
	# 
	# Future verisons, will most likely depreciate these
	# --------------------------------------------------
	def getFrame( self ):
		return self.dframe # this is primarly used for debugging

	def XY( self ):
		self.X = self.dframe.loc[ :, self.feature_range[0]:self.feature_range[1] ]
		self.Y = self.dframe.loc[:, self.predict ]
		return self
	# --------------------------------------------------


	def partition( self, train_validation_percent=0.75 ):

		self.X = self.dframe.loc[ :, self.feature_range[0]:self.feature_range[1] ]
		self.Y = self.dframe.loc[:, self.predict ]

		X_train, X_test, Y_train, Y_test = train_test_split( self.X, self.Y, 
															 test_size=(1 - train_validation_percent), 
															 random_state = None, stratify = self.Y )

		return X_train, X_test, Y_train, Y_test


	def parse( self, remove_columns=None ):

		# remove unwanted columns from the data
		self.dframe.drop( columns=remove_columns, axis=1, inplace=True )

		# remove subjects that have one, or more, NA values 
		for i in range( 4, len( self.dframe.columns ) ):
			self.dframe.drop( self.dframe[ self.dframe.iloc[:,i].isnull() ].index, axis=0, inplace=True )

		# remove subjects from the dataframe that are not HC (0) or the SDx prediction value
		self.dframe = self.dframe[ ( self.dframe[ self.predict ] == 0 ) | ( self.dframe[ self.predict ] == self.predict_val ) ].copy()

		# probably not necessary, but just in case, replace SDx value with one. Now the labels are just 0 and 1 (and not 0 and 3 or 4 or 5 or 6)
		self.dframe[ self.predict ].replace( self.predict_val, 1, inplace=True )

		if self.harmonize:

			# not 100% sure why batch ('site') has to be a covariate in neuroCombat, but it is required, hmmmm :/
			self.covariates.insert( 0, self.batch )
			
			self.dframe.loc[ :, self.feature_range[0]:self.feature_range[1] ] = neuroCombat( data=self.dframe.loc[ :, self.feature_range[0]:self.feature_range[1] ], 
																				   covars=self.dframe[ self.covariates ], batch_col=self.batch )

		return self	

		
