from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

class DenseClassificationModel:
    
    def __init__( self, input_dim=0 ):
        
        self.model = Sequential()
        self.input_dim = input_dim
        
        
        self.grid_dict = { 'classifier__learn_rate': [0.1, 0.15 ],
                           'classifier__epochs': [ 50, 150 ],
                           'classifier__hidden_units_L1': [ 10, 20 ],
                           'classifier__hidden_units_L2': [ 2, 5 ],
                           'classifier__l2_reg_penalty': [ 0.01, 0.1 ],
                           'classifier__drop_out_rate': [ 0.1, 0.2 ] }
        
        self.name = 'classifier'
        
    def get_name( self ):
        
        return self.name
        
    def set_input_dim( self, input_dim=0 ):
        
        self.input_dim = input_dim
        
    def construct( self, learn_rate=0.1, hidden_units_L1=5, 
                   hidden_units_L2=10, l2_reg_penalty=0.01, drop_out_rate=0.1, debug=False ):
                   
        self.dense_layers = []
        self.arch = []
        
        if debug:
            self.print( "lr={0:.6f}, hu_L1={1:d}, hu_L2={2:d}, l2_pen={3:.3f}".format( 
                         learn_rate, hidden_units_L1, hidden_units_L2, l2_reg_penalty ) )
        
        self.model.add( Dense( units=hidden_units_L1, 
                               activation='relu', 
                               input_shape=( self.input_dim, ) ) )
                               
        self.dense_layers.append( 0 )
        self.arch.append( self.input_dim )
        self.arch.append( hidden_units_L1 )
        
        self.model.add( Dropout( rate=drop_out_rate ) )
        
        self.model.add( Dense( units=hidden_units_L2, 
                               kernel_regularizer=l2( l2_reg_penalty ),
                               activation='relu' ) )
                               
        self.dense_layers.append( 2 )
        self.arch.append( hidden_units_L2 )
        
        self.model.add( Dense( units=1, 
                               activation='sigmoid' ) )
        
        self.model.compile( optimizer=Adam( lr=learn_rate ), 
                            loss='binary_crossentropy', 
                            metrics=['accuracy'] )
        #print( "here" )            
        #print( self.arch )
        #print( self.dense_layers )
        
        return self.model
    
    
    def set_grid_dict( self, grid_dict = { 'classifer__learn_rate': [ 0.01, 0.1 ] } ):
        
        self.grid_dict = grid_dict
        
    def get_grid_dict( self ):
        
        return self.grid_dict

    def get( self ):
        
        return self.model
        
    def select_features( self, cv_models, features, top_num_features = 20, weight_matrix_threshold = 0.5 ):
    
    	num_folds = len( cv_models['estimator'] )
    	
    	mask_matrices = [None]*( len( self.arch ) - 1 )
    	weight_matrices = [None]*( len( self.arch ) - 1 )
    	
    	for i in range( 0, ( len( self.arch ) - 1 ) ):
    	
    		mask_matrices[i] = np.zeros( ( self.arch[i], self.arch[i+1] ) )
    		
    		# print( mask_matrices[i].shape )
    	
    		for j in range( 0, num_folds ):
    		
    			mask_matrices[i] = mask_matrices[i] + cv_models['estimator'][i]['classifier'].model.get_weights()[ self.dense_layers[i] ]
    		
    		mask_matrices[i] = np.transpose( mask_matrices[i] )	
    		
    	col_idx = []
    	
    	# print( len( self.arch ) )
    	
    	for i in range( ( len( self.arch ) - 2 ), -1, -1 ):
    	
    		# print( i )
    	
    		idx = mask_matrices[i] < np.max( mask_matrices[i] )*weight_matrix_threshold
    		mask_matrices[i][ idx ] = 0
    		idx = mask_matrices[i] > 0
    		mask_matrices[i][ idx ] = 1
    		mask_matrices[i][ col_idx, : ] = 0
    		
    		z_vec = np.sum( mask_matrices[i], axis=0 )
    		col_idx = z_vec == 0
    		
    		plt.figure( figsize=[5.2,5.2])
    		plt.imshow( mask_matrices[i]  )
    		plt.set_cmap('hot')
    		plt.colorbar()
    		
    		
    	for i in range( 0, ( len( self.arch ) - 1 ) ):
    	
    		weight_matrices[i] = np.zeros( ( self.arch[i], self.arch[i+1] ) )
    	
    		for j in range( 0, num_folds ):
    		
    			weight_matrices[i] = weight_matrices[i] + np.multiply( np.transpose( mask_matrices[i] ), np.abs( cv_models['estimator'][i]['classifier'].model.get_weights()[ self.dense_layers[i] ] ) )
    		
    		weight_matrices[i] = np.transpose( weight_matrices[i] )	
    		
    	
    	weight_sum = np.sum( weight_matrices[ ( len( weight_matrices ) - 1 ) ], axis=0 )
    	
    	for i in range( ( len( weight_matrices ) - 2 ), -1, -1 ):
    	
    		# print( i )
    		# print( weight_matrices[i].shape )
    	
    		weight_matrices[i][:,0] = weight_matrices[i][:,0] + weight_sum
    		weight_sum = np.sum( weight_matrices[ i ], axis=0 )
    	
    	idx = np.argsort( weight_sum * -1 )
    	
    	for i in range( 0, top_num_features ):
    		print( "{0} (w={1}) ".format( features[idx[i]], weight_sum[idx[i]] ) )
    		
    	return idx, weight_sum	
	
