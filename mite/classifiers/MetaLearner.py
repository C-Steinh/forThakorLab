import numpy as np
from sklearn.preprocessing import normalize
from ..utils.DynamicKDMap import DynamicKDMap
class MetaLearner:
    def __init__(self, X, y, S, model = 'SRC', minmax_size = ( 50, 10000 ) ):
        # normalize input data
        X = normalize( X, norm = 'l2', axis = 1 )

        # import the appropriate model
        try:
            exec( "from . import " + model )
        except ImportError:
            model = 'SRC'
            exec( "from . import " + model )
            print( 'Invalid classifier model. Changing to SRC!', flush = True )
        self.__mdl = eval( model )( X, y )

        # create class sets
        self.__kdmaps = []
        self.__classes = np.unique( y )
        self.__minmax_size = minmax_size
        self.__sim_thresh = np.zeros( len( self.__classes ) )
        for c in range( 0, len( self.__classes ) ):
            idx = np.isin( y, self.__classes[ c ] )
            self.__kdmaps.append( DynamicKDMap( leafsize = 100 ) )  # empty KD tree
            
            if np.any( idx ):
                self.__kdmaps[ c ].insert( S[ idx, : ], X[ idx, : ] )               # add states and features to KD maps
                iclass = X[ idx, : ]
                oclass = X[ ~idx, : ]
                isim = ( iclass @ iclass.T ).flatten()
                osim = ( oclass @ iclass.T ).flatten()
                self.__sim_thresh[ c ] = 0.5 * ( np.amin( isim ) + np.amax( osim ) )

    def predict( self, X, S, knn = 50, update = True, prob = False ):
        if len( X.shape ) == 1: 
            X.resize( ( 1, X.shape[0] ) )
            S.resize( ( 1, S.shape[0] ) )
        X = normalize( X, norm = 'l2', axis = 1 )

        yhat = np.zeros( X.shape[0], dtype = np.int )
        yprob = np.zeros( ( X.shape[0], len( self.__classes ) ), dtype = np.float )
        for sample in range( 0, X.shape[0] ):
            # grab sample data
            x = np.expand_dims( X[ sample, : ], axis = 0 )
            s = np.expand_dims( S[ sample, : ], axis = 0 )

            # update the training data
            Xtrain = []
            ytrain = []
            for c in range( 0, len( self.__classes ) ):
                cs, cx = self.__kdmaps[ c ].search( s, knn )     # get our nearest neighbors [may not be total amount though]
                Xtrain.append( cx )
                ytrain.append( c * np.ones( cs.shape[0] ) )
            Xtrain = np.vstack( Xtrain )
            ytrain = np.hstack( ytrain )

            # retrain model and classify
            self.__mdl.train( Xtrain, ytrain )
            yhat[ sample ], yprob[ sample, : ] = self.__mdl.predict( x, prob = prob )

            # we are dynamically updating the model
            if update:
                simval = np.median( ( Xtrain[ np.isin( ytrain, yhat[ sample ] ), : ] @ x.T ) )
                if simval > self.__sim_thresh[ yhat[sample] ]:                              # similarity value greater than threshold
                    self.__kdmaps[ yhat[ sample ] ].insert( s, x )                          # add state / feature to appropriate tree
                    if self.__kdmaps[ yhat[ sample ] ].size > self.__minmax_size[ 1 ]:      # this class size is too big
                        self.__kdmaps[ yhat[ sample ] ].shrink( self.__minmax_size[ 0 ] )   # shrink it
        return yhat, yprob

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from ..utils.Metrics import confusion_matrix

    import time
    import matplotlib.pyplot as plt

    data = load_digits()
    Xtrain, Xtest, ytrain, ytest = train_test_split( data['data'], data['target'], test_size = 0.33 )
    Strain = np.random.rand( ytrain.shape[0], 100 )
    Stest = np.random.rand( ytest.shape[0], 100 )

    t1 = time.clock()
    mdl = MetaLearner( Xtrain, ytrain, Strain, model = 'SRC' )
    t2 = time.clock()
    print( "Training time:", 1000 * ( t2 - t1 ), "ms" )

    t = np.zeros( ytest.shape )
    yhat_dyn = np.zeros( ytest.shape )
    for idx in range( 0, np.size( ytest ) ):
        t1 = time.clock()
        yhat_dyn[ idx ], _ = mdl.predict( Xtest[ idx, : ], Stest[ idx, : ], knn = 50 )
        t2 = time.clock()
        t[ idx ] = 1000 * ( t2 - t1 )

    print( "Total Testing Time:", np.sum( t ), "ms" )
    print( "Mean Testing Time:", np.mean( t ), "ms" )
    print( "Max Testing Time:", np.amax( t ), "ms" )

    confusion_matrix( ytest, yhat_dyn, labels = data['target_names'], cmap = 'Blues' )