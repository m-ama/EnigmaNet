import os
import logging
import numpy as np
import pandas as pd
import transformers as trans
from sklearn.model_selection import train_test_split

class enigma_data(object):
    """
    The enigmanet object contains all functions and information of
    training and validating an AI model for epilepsy prediction. Reads
    input Excel path into a Pandas dataframe and performs initial
    integrity check. Data is loaded into the object itself as
    enigmanet.dFrame
    """
    
    def __init__(self,
                 path=None,
                 classcol=None,
                 sitecol=None,
                 dbegin=None,
                 dend=None,
                 fillmissing=True,
                 harmonize=False,
                 crange=None,
                 batch=None,
                 discrete=None,
                 continuous=None):
        
        self.DEBUG = False
        
        if os.path.exists(path):
            self.dframe = pd.read_csv(path)  # Create Dataframe
            # Integrity check
            if self.DEBUG:
                print( 'Found classes: ' + str(self.dframe.loc[:,classcol].unique()))
                
        else:
            assert isinstance(path, object)
        if classcol is None:
            raise ValueError('Please define the column containing class '
                             'information')
        else:
            self.classcol = classcol
        if sitecol is None:
            logging.warning('Site column undefined, cannot perform data '
                            'harmonization')
        self.classcol = classcol
        self.sitecol = sitecol
        self.drange = [dbegin, dend]
        self.fillmissing = fillmissing
        
        if harmonize:
            self.harmonize = True
            self.crange = crange
            self.batch = batch
            self.discrete = discrete
            self.continuous = continuous
            
    
    def XY( self ):
        self.X = self.dframe.loc[:, self.drange[0]:self.drange[1]]
        self.Y = self.dframe.loc[:, self.classcol]
        return self
    
    def partition( self, train_validation_percent=0.75 ):
        self.XY()
        X_train, X_test, Y_train, Y_test = train_test_split( self.X, self.Y,
                                                             test_size=(1 - train_validation_percent), 
                                                             random_state = None,
                                                             stratify = self.Y )
        return X_train, X_test, Y_train, Y_test
        
        

    def transform(self):
        """Applies transformations to data"""
        if self.fillmissing:
            self.dFrame = trans.classfill(dFrame=self.dframe,
                                          classCol=self.classcol,
                                          siteCol=self.sitecol,
                                          idxRange=self.drange)
        else:
            if self.DEBUG:
                print('Skipping fillmissing')
            
        if self.harmonize:
            self.dframe = trans.combat(dFrame=self.dframe,
                                       drange=self.drange,
                                       crange=self.crange,
                                       batch=self.batch,
                                       discrete=self.discrete,
                                       continuous=self.continuous)
        else:
            if self.DEBUG:
                print('Skipping ComBat harmonization')
            
        return self
