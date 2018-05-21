import numpy as np
import numpy.matlib
from numpy import linalg as la
from sklearn.preprocessing import normalize

import sparsesolvers as ss

class SRC:
    def __init__(self, X, y, tol = 0.1 ):
        self.__dictionary = None    # X
        self.__labels = None        # y
        self.__tolerance = tol

        self.__class_vector = np.unique( y )
        self.__num_classes = np.size( self.__class_vector )

        self.train( X, y )

    def train(self, X, y):
        self.__dictionary = normalize( X.T, norm = 'l2', axis = 0 )
        self.__labels = y
    
    def predict(self, X, prob = False):
        X = normalize( X.T, norm = 'l2', axis = 0 )
        residuals = np.zeros( ( X.shape[1], self.__num_classes ), dtype = np.float )
        for sample in range( 0, X.shape[1] ):
            x = X[ :, sample ]
            # solve the L1 minimization problem
            s, _ = ss.Homotopy( self.__dictionary ).solve( x, tolerance = self.__tolerance )

            # reconstruction residuals
            for ind in range( 0, self.__num_classes ):
                c = self.__class_vector[ ind ]
                coef_c = s[ np.equal( self.__labels, c ) ]
                Dc = self.__dictionary[ :, np.equal( self.__labels, c ) ]
                residuals[ sample, ind ] = la.norm( x - Dc @ coef_c, ord = 2 )

        predicts = self.__class_vector[ np.argmin( residuals, axis = 1 ) ]
        probs = np.divide( 1.0 - residuals, np.sum( 1.0 - residuals, axis = 1 )[ :, None ] ) if prob else None # TODO: LOOK UP BETTER METRIC
        return predicts, probs

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from ..utils.Metrics import confusion_matrix

    import time

    data = load_digits()
    Xtrain, Xtest, ytrain, ytest = train_test_split( data['data'], data['target'], test_size = 0.33 )

    t1 = time.clock()
    mdl = SRC( Xtrain, ytrain )
    t2 = time.clock()
    yhat, _ = mdl.predict( Xtest, prob = True )
    t3 = time.clock()

    print( 'Training time:', 1000 * ( t2 - t1 ), 'ms' )
    print( 'Testing time:', 1000 * ( t3 - t2 ), 'ms' )

    cm = confusion_matrix( ytest, yhat, labels = data['target_names'] )