from enum import Enum

from DenseClassificationModel import DenseClassificationModel
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import Constants as constants

class Model_Factory:
    
    def build( self, model_type=constants.MODELS.DL_CLASSIFY, input_dim=0 ):
        
        if model_type == constants.MODELS.SV_CLASSIFY:
            
            return None #working on this while the DL classification results are running
        
        elif model_type == constants.MODELS.DL_CLASSIFY:
            
            return DenseClassificationModel( input_dim )
        
        elif model_type == constants.MODELS.SCALER:
            
            return MinMaxScaler()
        
        else:
            
            return SMOTE()
        
    
    
