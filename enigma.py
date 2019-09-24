import os
import logging
import numpy as np
import pandas as pd
from transformations import enigmatransforms as trans

class enigmanet(object):
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
                 continuous=None,
                 scale=True,
                 testsplit=0.10):
        if os.path.exists(path):
            self.dframe = pd.read_csv(path)  # Create Dataframe
            # Integrity check
            print(
                'Found classes: ' + str(self.dframe.loc[:,
                                        classcol].unique()))
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
        self.scale = scale
        self.testsplit = testsplit
        if harmonize:
            self.harmonize = True
            self.crange = crange
            self.batch = batch
            self.discrete = discrete
            self.continuous = continuous

    def transformdata(self):
        """Applies transformations to data"""
        if self.fillmissing:
            self.dFrame = trans.classfill(dFrame=self.dframe,
                                          classCol=self.classcol,
                                          siteCol=self.sitecol,
                                          idxRange=self.drange)
        else:
            print('Skipping fillmissing')
        if self.harmonize:
            self.dframe = trans.combat(dFrame=self.dframe,
                                       drange=self.drange,
                                       crange=self.crange,
                                       batch=self.batch,
                                       discrete=self.discrete,
                                       continuous=self.continuous)
        else:
            print('Skipping ComBat harmonization')
        if self.scale:
            self.dframe = trans.scale(dFrame=self.dframe,
                                      drange=self.drange)
        else:
            print('Skipping data scaling')

    def splitdata(self, oversample=True):
        x_train, x_test, y_train, y_test = trans.split(
            dFrame=self.dframe,
            classCol=self.classcol,
            drange=self.drange,
            datasplit=self.testsplit)
        if oversample:
            x_train, y_train = trans.oversample(x_train, y_train)
        return x_train, x_test, y_train, y_test