import os
import logging
from transformations import enigmatransforms as trans



class enigmanet(object):
    """
    The enigmanet object contains all functions and information of
    training and validating an AI model for epilepsy prediction.
    """
    def __init__(self,
                 path,
                 classcol=None,
                 sitecol=None,
                 dbegin=None,
                 dend=None,
                 fillmissing=True,
                 harmonize=True,
                 scale=True,
                 splitdata=0.10):
        if os.path.exists(path):
            self.dFrame = pd.read_csv(path)  # Create Dataframe
        else:
            assert isinstance(imPath, object)
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
        self.drange = [dBegin, dEnd]
        self.fillmissing = fillmissing
        self.scale = scale
        self.splitdata = splitdata

    def tf