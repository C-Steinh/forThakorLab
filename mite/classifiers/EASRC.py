import numpy as np
import numpy.matlib
from numpy import linalg as la
from sklearn.preprocessing import normalize

import sparsesolvers as ss
from hpelm import ELM
class EASRC:
    def __init__(self, X, y, lam = 0.01, alpha = 0.1, tol = 0.1, elm_method = 'ELM', nn_hidden_size = 1000 ):
        self.__dictionary = None    # X
        self.__labels = None        # y
        self.__class_vector = np.unique( y )
        self.__num_classes = np.size( self.__class_vector )
        self.__kclass = int( np.size( np.unique( y ) ) / 2 )

        self.__lambda = lam
        self.__alpha = alpha
        self.__tolerance = tol

        self.__elm = None
        self.__elm_width = nn_hidden_size

        # self.__net_params = { 'ActiveFunction'  : 'sigmoid',
        #                       'InputSize'       : X.shape[1],
        #                       'HiddenSize'      : nn_hidden_size,
        #                       'Method'          : elm_method,
        #                       'Type'            : 'classification',
        #                       'Bias'            : None,
        #                       'Weight'          : None, 
        #                       'C'               : np.exp( np.linspace( -4.0, 4.0, 41 ) ),
        #                       'C_opt'           : None,
        #                       'Beta'            : None,
        #                       'L00'             : None,
        #                       'TrainLabel'      : None,
        #                       'AccTrain'        : None }
        self.train( X, y )

    def train(self, X, y):
        self.__dictionary = normalize( X.T, norm = 'l2', axis = 0 )
        self.__labels = y
        
        # self.__elm_init()
        # self.__elm_train( self.__dictionary, EASRC.__convert_labels( y ) )

        self.__elm = ELM( X.shape[1], self.__num_classes, 
                          classification = "c", precision = 'single',
                          accelerator = None )
        self.__elm.add_neurons( self.__elm_width, 'sigm' )
        self.__elm.train( self.__dictionary.T, EASRC.__convert_labels( y ).T, 'LOO' )
    
    def predict(self, X, prob = False):
        X = normalize( X.T, norm = 'l2', axis = 0 ) # may end up flipping this
        elm_yhat = self.__elm.predict( X.T ).T
        # elm_yhat, _ = self.__elm_test( X )
        max_ind = np.argmax( elm_yhat, axis = 0 )
        Tf = np.amax( elm_yhat, axis = 0 )
        elm_yhat[ max_ind, range(0, max_ind.shape[0]) ] = -np.inf
        Ts = np.amax( elm_yhat, axis = 0 )
        Tdiff = Tf - Ts
        
        predicts = np.zeros( X.shape[1], dtype = np.int )
        elm_only = Tdiff > self.__alpha
        predicts[ elm_only ] = max_ind[ elm_only ]

        for sample in np.where( np.logical_not( elm_only ) )[0]:
            x = X[ :, sample ]
            slabel = np.argsort( elm_yhat )
            newidx = np.isin( self.__labels, slabel[ :self.__kclass ] )
            newtrainlabel = self.__labels[ newidx ]
            newdict = self.__dictionary[ :, newidx ]
            
            # solve the L1 minimization problem
            s, _ = ss.Homotopy( newdict ).solve( x, tolerance = self.__tolerance )
            newlabel = np.unique( newtrainlabel )

            # reconstruction residuals
            residual = np.zeros( np.size( newlabel ) )
            sum_c = np.zeros( np.size( newlabel ) )
            for ind in range( 0, np.size( newlabel ) ):
                coef_c = s[ np.equal( newtrainlabel, newlabel[ ind ] ) ]
                Dc = newdict[ :, np.equal( newtrainlabel, newlabel[ ind ] ) ]
                residual[ ind ] = np.square( la.norm( x - np.dot( Dc, coef_c ) ) )
                sum_c[ ind ] = np.sum( coef_c )

            # predict1 = newlabel[ np.argmin( residual ) ]
            predicts[ sample ] = newlabel[ np.argmin( residual ) ]

        return self.__class_vector[ predicts ], None

    def __elm_init( self, scale = 1 ):
        self.__net_params['Bias'] = 2 * scale * np.random.rand( self.__net_params['HiddenSize'], 1 ) - scale
        self.__net_params['Weight'] = 2 * scale * np.random.rand( self.__net_params['HiddenSize'], self.__net_params['InputSize'] ) - scale

    def __elm_train(self, X, y):
        ndata = X.shape[1]
        tmpH = np.dot( self.__net_params['Weight'], X ) + np.matlib.repmat( self.__net_params['Bias'], 1, ndata )

        if self.__net_params['ActiveFunction'].lower() == 'sigmoid':
            H = np.divide( 1, 1 + np.exp( -tmpH ) )
        elif self.__net_params['ActiveFunction'].lower() == 'tanh':
            H = np.tanh( tmpH )
        else: raise RuntimeError( 'Invalid activation function!' )

        if self.__net_params['Method'].upper() == 'ELM':
            beta, C_opt, LOO = EASRC.__regressor( H.T, y.T, 0 )
        elif self.__net_params['Method'].upper() == 'SRELM':
            beta, C_opt, LOO = EASRC.__regressor( H.T, y.T, np.mean( self.__net_params['C'] ) )
        elif self.__net_params['Method'].upper() == 'RELM':
            beta, C_opt, LOO = EASRC.__regressor( H.T, y.T, self.__net_params['C'] )
        else: raise RuntimeError( 'Invalid regression method!' )

        self.__net_params['C_opt'] = C_opt
        self.__net_params['LOO'] = LOO
        self.__net_params['Beta'] = beta.T

        yhat = np.dot( self.__net_params['Beta'], H )

        if self.__net_params['Type'].lower() == 'classification':
            label_actual = np.argmax( yhat, axis=0 )
            label_desired = np.argmax( y, axis = 0 )
            acc_train = np.sum( np.equal( label_actual, label_desired ) ) / ndata
        elif self.__net_params['Type'].lower() == 'regression':
            fronorm = la.norm( y - yhat, ord='fro' )
            acc_train = np.sqrt( np.square( fronorm ) / ndata )
        else: raise RuntimeError( 'Invalid network type!' )
        return acc_train

    def __elm_test( self, X, y = None ):
        ndata = X.shape[1]
        tmpH = np.dot( self.__net_params['Weight'], X ) + np.matlib.repmat( self.__net_params['Bias'], 1, ndata )

        if self.__net_params['ActiveFunction'].lower() == 'sigmoid':
            H = np.divide( 1, 1 + np.exp( -tmpH ) )
        elif self.__net_params['ActiveFunction'].lower() == 'tanh':
            H = np.tanh( tmpH )
        else: raise RuntimeError( 'Invalid activation function!' )

        yhat = np.dot( self.__net_params['Beta'], H )
        acc_test = None
        if self.__net_params['Type'].lower() == 'classification':
            label_actual = np.argmax( yhat, axis=0 )
            if y is not None:
                label_desired = np.argmax( yhat, axis=0 )
                acc_test = np.sum( np.equal( label_actual, label_desired ) ) / ndata
        elif self.__net_params['Type'].lower() == 'regression':
            if y is not None:
                fronorm = la.norm( y - yhat, ord='fro' )
                acc_test = np.sqrt( np.square( fronorm ) / ndata )
        else: raise RuntimeError( 'Invalid network type!' )
        # self.__net_params['TestLabel'] = yhat
        return yhat, acc_test

    def __regressor( H, y, lambdas = np.exp( np.linspace( -7, 7, 141 ) ) ):
        ndata = H.shape[0]

        if np.size( lambdas ) == 1: # if scalar lambda
            opt_lambda = lambdas
            LOO = np.inf
            if ndata < H.shape[1]:
                tmp = np.dot( H, H.T ) + opt_lambda * np.eye( ndata )
                beta = np.dot( np.dot( H.T, la.pinv( tmp ) ), y )
            else:
                tmp = np.dot( H.T, H ) + opt_lambda * np.eye( H.shape[1] )
                beta = np.dot( np.dot( la.pinv( tmp ), H.T ), y )
        else:
            LOO = np.inf * np.ones( np.size( lambdas ) )
            if ndata < H.shape[1]:
                HH = np.dot( H, H.T )
                U, S, _ = la.svd( HH, full_matrices = False )
                A = np.dot( HH, U )
                B = np.dot( U.T, y )
                for i in range( 0, np.size( lambdas ) ):
                    l = lambdas[ i ] 
                    tmp = np.multiply( A, np.matlib.repmat( np.divide( 1, S + l ), np.size( S ), 1 ) ) 
                    HAT = np.sum( np.multiply( tmp, U ), axis=1 )
                    yhat = np.dot( tmp, B )
                    errdiff = np.divide( y - yhat, np.matlib.repmat( 1 - HAT, y.shape[1], 1 ).T )
                    fronorm = la.norm( errdiff, ord='fro' )
                    LOO[ i ] = np.square( fronorm ) / ndata
                ind = np.argmin( LOO )
                opt_lambda = lambdas[ ind ]
                tmp = np.multiply( U, np.matlib.repmat( np.divide( 1, S + opt_lambda ), np.size( S ), 1 ) )
                beta = np.dot( np.dot( H.T, tmp ), B )
            else:
                U, S, _ = la.svd( np.dot( H.T, H ) )
                A = np.dot( H, U )
                B = np.dot( A.T, y )
                for i in range(0, np.size( lambdas ) ):
                    l = lambdas[ i ]
                    tmp = np.multiply( A, np.matlib.repmat( np.divide( 1, S + l ), A.shape[0], 1 ) )
                    HAT = np.sum( np.multiply( tmp, A ), axis=1 )
                    yhat = np.dot( tmp, B )
                    errdiff = np.divide( y - yhat, np.matlib.repmat( ( 1 - HAT ), y.shape[1], 1 ).T )
                    fronorm = la.norm( errdiff, ord='fro' )
                    LOO[ i ] = np.square( fronorm ) / ndata
                ind = np.argmin( LOO )
                opt_lambda = lambdas[ ind ]
                tmp = np.matlib.repmat( np.divide( 1, S + opt_lambda ), np.size( S ), 1 )
                beta = np.dot( np.multiply( U, tmp ), B )
        
        return beta, opt_lambda, LOO

    def __convert_labels( y ):
        classes = np.unique( y )
        n_classes = np.size( classes )
        n_data = np.size( y )

        y_new = -1 * np.ones( ( n_classes, n_data ), dtype = np.float )
        for i in range( 0, n_classes ):
            y_new[ i, y == classes[ i ] ] = 1
        y_new = ( y_new + 1 ) / 2
        return y_new

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from ..utils.Metrics import confusion_matrix

    import time

    data = load_digits()
    Xtrain, Xtest, ytrain, ytest = train_test_split( data.data, data.target, test_size = 0.33 )

    t1 = time.clock()
    mdl = EASRC( Xtrain, ytrain )
    t2 = time.clock()
    yhat, _ = mdl.predict( Xtest, prob = True )
    t3 = time.clock()

    print( 'Training time:', 1000 * ( t2 - t1 ), 'ms' )
    print( 'Testing time:', 1000 * ( t3 - t2 ), 'ms' )

    cm = confusion_matrix( ytest, yhat, labels = data.target_names )