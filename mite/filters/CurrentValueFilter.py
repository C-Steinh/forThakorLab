import numpy as np

class CurrentValueFilter:
    def __init__(self):
        """ Constructor """
        pass

    def filter(self, raw):
        """ Filters raw data """
        return raw[-1, :]

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 )
    filt = CurrentValueFilter()
    features = filt.filter( data )
    print( features )
