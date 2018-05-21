import numpy as np

class TimeDomainFilter:
    def __init__(self, num_features = 5, eps_zc = 0.01, eps_ssc = 0.01):
        assert ( num_features == 3 or num_features == 5 )
        self.__num_features = num_features
        self.__eps_zc = eps_zc
        self.__eps_ssc = eps_ssc

    def filter(self, raw):
        num_channels = raw.shape[1] if raw.ndim == 2 else raw.shape[0]
        feat = np.zeros( self.__num_features * num_channels )
        for chan in range( 0, num_channels ):
            idx = chan * self.__num_features
            feat[ idx ] = np.sum( np.abs( raw[ :, chan ] ) )
            feat[ idx + 1 ] = np.var( raw[ :, chan ] )
            feat[ idx + 2 ] = np.sum( np.abs( raw[ :-1, chan ] - raw[ 1:, chan ] ) )
            
            if self.__num_features == 5:
                zc = np.vstack( [ np.abs( raw[ :-1, chan ] ), np.abs( raw[ 1:, chan ] ),
                                np.multiply( raw[ :-1, chan ], raw[ 1:, chan ] ) ] )
                feat[ idx + 3 ] = np.sum( np.logical_and( zc[ 2, : ] < 0, zc[ 1, : ] > self.__eps_zc, zc[ 0, : ] > self.__eps_zc ) )

                chan_dt = np.gradient( raw[ :, chan ] )
                ssc = np.vstack( [ np.abs( chan_dt[ :-1 ] ), np.abs( chan_dt[ 1: ] ),
                                 np.multiply( chan_dt[ :-1 ], chan_dt[ 1: ] ) ] )
                feat[ idx + 4 ] = np.sum( np.logical_and( ssc[ 2, : ] < 0, ssc[ 1, : ] > self.__eps_ssc, ssc[ 0, : ] > self.__eps_ssc ) )
        return feat

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 )
    filt = TimeDomainFilter( num_features = 5 )
    features = filt.filter( data )
    print( features.shape )