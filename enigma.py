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
                 classcol=None,
                 sitecol=None,
                 dbegin=None,
                 dend=None,
                 fillmissing=True,
                 harmonize=True,
                 scale=True,
                 splitdata=0.10):
        if os.path.exists(path):
            self.dframe = pd.read_csv(path)  # Create Dataframe
            # Integrity check
            print(
                'Found classes: ' + str(dFrame.loc[:, classcol].unique()))
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
        self.splitdata = splitdata

    def transformdata(self):
        """Applies transformations to data"""
        if self.fillmissing:
            self.dFrame = trans.classfill(self.dFrame,
                                          self.classCol,
                                          self.siteCol,
                                          [drange[0], drange[1]])
