import numpy as np

class FourierTransformFilter:
    def __init__(self, fftlen = 256):
        """ Constructor """
        self.__fftlen = fftlen

    def filter(self, raw):
        """ Filters raw data """
        feat = np.fft.fft(raw, n = self.__fftlen, axis = 0)
        return np.abs( feat[0:int(self.__fftlen/2.0)] )

if __name__ == '__main__':
    data = np.random.rand( 1024, 8 )
    filt = FourierTransformFilter( fftlen = 256 )
    features = filt.filter( data )
    print( features )