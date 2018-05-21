from sklearn.preprocessing import MinMaxScaler

class MinMaxFilter:
    def __init__(self, X, dynamic = False):
        self.__dynamic = dynamic
        self.__scalar = MinMaxScaler( feature_range = ( 0, 1 ), copy = False )
        self.__scalar.fit( X )

    def filter( self, raw ):
        if self.__dynamic: self.__scalar.partial_fit( raw )
        return self.__scalar.transform( raw )

if __name__ == "__main__":
    import numpy as np

    X = 10 * np.random.rand( 1000, 40 )
    filt = MinMaxFilter( X )
    f = filt.filter( 10 * np.random.rand( 10, 40 ) )

    print( f )