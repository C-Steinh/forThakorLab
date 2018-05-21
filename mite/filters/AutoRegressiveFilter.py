import numpy as np
import spectrum as sp

class AutoRegressiveFilter:
    def __init__(self, order = 6):
        """ Constructor """
        self.__order = order

    def filter(self, raw):
        """ Filters raw data """
        n_channels = raw.shape[1]
        feat = np.zeros( n_channels * self.__order )
        for chan in range( 0, n_channels ):
            idx1 = chan * self.__order
            idx2 = idx1 + self.__order
            feat[ idx1:idx2 ] = np.abs( sp.arburg( raw[ :, chan ], self.__order )[0] )
        return feat

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 )
    filt = AutoRegressiveFilter( order = 6 )
    features = filt.filter( data )
    print( features )