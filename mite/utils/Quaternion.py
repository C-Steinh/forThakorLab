import numpy as np
import numpy.linalg as la

def magnitude( q ):
    '''Return the magnitude of the quaternion'''
    return np.sqrt( np.sum( np.square( q ) ) )

def normalize( q ):
    '''Return the normalized quaternion'''
    return np.divide( q, magnitude( q ) )

def conjugate( q ):
    '''Return the conjugate of the quaternion'''
    return np.multiply( q, np.array( [ 1, -1, -1 , -1 ] ) )

def inverse( q ):
    '''Return the inverse of the quaternion'''
    return normalize( conjugate( q ) )

def multiply( q1, q2 ):
    '''Return the Hamilton product of the two quaternions'''
    qm = np.zeros( q1.shape, dtype = np.float )
    qm[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    qm[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    qm[2] = q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3]
    qm[3] = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1] 
    return qm

def average( q_all, axis = 1 ):
    '''Average the given set of quaternions'''
    n_quats = q_all.shape[axis]
    Q = np.zeros( ( 4, 4 ), dtype = np.float )
    for sample in range(0, n_quats):
        q = q_all[:, sample] if axis == 1 else q_all[sample,:]
        Q = np.outer( q, q ) + Q
    Q = np.divide( Q, n_quats )
    vals, vecs = la.eig( Q )
    return vecs[ :, np.argmax( vals ) ]

def relative( src, dest ):
    '''Compute the destination quaternion relative to the source'''
    return normalize( multiply( inverse( src ), dest ) )

def rotate( q, p ):
    '''Rotate the vector p by the quaternion q'''
    pp = np.zeros( q.shape, dtype = np.float )
    pp[1:] = p
    qp = inverse( q )
    pr = multiply( multiply( q, pp ), qp )
    return pr[1:]

def to_euler( q ):
    '''Compute the XYZ intrinsic Euler angles from the quaternion'''
    test = 2*(q[0]*q[2] - q[3]*q[1])
    test = 1.0 if test > 1.0 else test
    test = -1.0 if test < -1.0 else test
    
    phi = np.arctan2( 2*(q[0]*q[1]+q[2]*q[3]), 
                      1 - 2*(q[2]*q[2]+q[3]*q[3]) )
    theta = np.arcsin( test )
    psi = np.arctan2( 2*(q[0]*q[3]+q[1]*q[2]), 
                      1 - 2*(q[2]*q[2]+q[3]*q[3]) )
    return np.array( [phi, theta, psi] )

def from_euler( angles ):
    '''Compute the quaternion from the XYZ intrinsic Euler angles'''
    halfphi = 0.5 * angles[0]
    halftheta = 0.5 * angles[1]
    halfpsi = 0.5 * angles[2]
    
    q = np.zeros( 4, dtype = np.float )
    q[0] = np.cos( halfphi ) * np.cos( halftheta ) * np.cos( halfpsi ) \
         + np.sin( halfphi ) * np.sin( halftheta ) * np.sin( halfpsi )
    q[1] = np.sin( halfphi ) * np.cos( halftheta ) * np.cos( halfpsi ) \
         - np.cos( halfphi ) * np.sin( halftheta ) * np.sin( halfpsi )
    q[2] = np.cos( halfphi ) * np.sin( halftheta ) * np.cos( halfpsi ) \
         + np.sin( halfphi ) * np.cos( halftheta ) * np.sin( halfpsi )
    q[3] = np.cos( halfphi ) * np.cos( halftheta ) * np.sin( halfpsi ) \
         - np.sin( halfphi ) * np.sin( halftheta ) * np.cos( halfpsi )

    return q
    
def to_matrix( q ):
    '''Compute the rotation matrix from given quaternion'''
    R = np.zeros( (3, 3), dtype = np.float )
    
    R[0,0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
    R[0,1] = 2 * ( q[1]*q[2] - q[0]*q[3] )
    R[0,2] = 2 * ( q[0]*q[2] + q[1]*q[3] )

    R[1,0] = 2 * ( q[1]*q[2] + q[0]*q[3] )
    R[1,1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3]
    R[1,2] = 2 * ( q[2]*q[3] - q[0]*q[1] )

    R[2,0] = 2 * ( q[1]*q[3] - q[0]*q[2] )
    R[2,1] = 2 * ( q[0]*q[1] + q[2]*q[3] )
    R[2,2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]

    return R

def from_matrix( R ):
    '''Compute the quaternion from the given rotation matrix'''
    assert( R.shape == (3,3) )
    assert( la.det(R) == 1 )
    assert( np.allclose( np.dot( R.T, R ), np.eye(3) ) )
    
    q = np.zeros( 4 , dtype = np.float )
    q[0] = 0.5 * np.sqrt( 1 + R[0,0] + R[1,1] + R[2,2] )
    q[1] = ( 1 / ( 4 * q[0] ) ) * ( R[2,1] - R[1,2] )
    q[2] = ( 1 / ( 4 * q[0] ) ) * ( R[0,2] - R[2,0] )
    q[3] = ( 1 / ( 4 * q[0] ) ) * ( R[1,0] - R[0,1] )

    return q