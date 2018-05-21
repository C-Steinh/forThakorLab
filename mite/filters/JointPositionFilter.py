import numpy as np
from ..utils import Quaternion as quat

class JointPositionFilter:
    LIMB_VECTORS = [ np.array( [ 0,    0.10,  0.6000 ] ),
                     np.array( [ 0,    0.00, -0.3429 ] ),
                     np.array( [ 0,    0.00, -0.3048 ] ),
                     np.array( [ 0,    0.00, -0.0600 ] ) ]
    def __init__(self, num_quaternions = 5):
        num_quaternions = int( num_quaternions )
        assert( num_quaternions > 1 and num_quaternions < 6 )

        self.__num_quaternions = num_quaternions
        self.__joint_offset = 5 - self.__num_quaternions 

    def filter(self, raw):
        if len(raw.shape) == 1: raw = np.expand_dims( raw, axis=0 )
        assert( ( raw.shape[1] / 4 ) == self.__num_quaternions )
        num_samples = raw.shape[0]
        xyz = np.zeros( ( num_samples, 3 *( self.__num_quaternions - 1 ) ) )
        for i in range( 0, num_samples ):
            v = np.zeros( 3 )
            limbs = JointPositionFilter.LIMB_VECTORS[ -(self.__num_quaternions-1): ]
            for j in range( 1, self.__num_quaternions ):
                qidx = 4 * j
                qr = quat.relative( raw[ i, qidx-4:qidx ], raw[ i, qidx:qidx+4 ] ) 
                for k in range( j - 1, len( limbs ) ):
                    limbs[ k ] = quat.rotate( qr, limbs[ k ] )
                jidx = 3 * ( j - 1 )
                xyz[ i, jidx:jidx+3 ] = np.sum( np.vstack( limbs[ :j ] ), axis = 0 )
        return np.squeeze( xyz )

if __name__ == '__main__':
    n_quats = 5
    vals = np.random.rand( 1000, 4 * n_quats )
    # normalize
    for n in range(0, vals.shape[0]):
        for q in range( 0, n_quats ):
            qidx = 4 * q
            vals[ n, qidx:qidx+4 ] = quat.normalize( vals[ n, qidx:qidx+4 ] )
    
    filt = JointPositionFilter(num_quaternions=n_quats)
    feat = filt.filter( vals )
    print( feat )