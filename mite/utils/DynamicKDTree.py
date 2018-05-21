import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

class DynamicKDTree:
    @staticmethod
    def __default_distance_metric( y, X, axis = 1 ):
        return np.sum( np.square( np.subtract( X, y ) ), axis = axis )

    def __init__(self, leafsize = 100):
        self.__leafsize = leafsize
        self.__subtrees = [ None ]
        self.__size = 0

    def insert( self, pts ):
        if len( pts.shape ) == 1: pts = np.expand_dims( pts, 0 )
        self.__size += pts.shape[0]
        for pt in pts:
            idx = self.__subtrees.index( None ) if None in self.__subtrees else -1
            if idx == -1: 
                self.__subtrees.append( None )
                idx += len( self.__subtrees )

            all_pts = [ pt ]
            for i in range( idx - 1, -1, -1 ):
                all_pts.append( self.__subtrees[ i ].points )
                self.__subtrees[ i ] = None
            all_pts = np.vstack( [ x for x in all_pts if len( x ) ] ) # gets rid of empy lists
            self.__subtrees[ idx ] = StaticKDTree( all_pts, self.__leafsize )
    
    def delete( self, q ):
        if len( q.shape ) == 1: q = np.expand_dims( q, 0 )
        self.__size = 0
        for tree in self.__subtrees:
            if tree is not None: 
                tree.delete( q )
                self.__size += tree.size

    def search( self, q, k, dfunc = None ):
        if dfunc is None: dfunc = DynamicKDTree.__default_distance_metric
        nn = np.inf * np.ones( ( k, np.size( q ) ) )
        dmax = np.inf
        for tree in self.__subtrees:
            if tree is not None: nn, dmax = tree.search( q, k, nn, dmax, dfunc )
        if nn is not None: return nn[ ~np.isinf( nn ).any( axis = 1 ) ]

    def reset( self ):
        self.__subtrees = [ None ]
        self.__size = 0

    def show( self ):
        fig = plt.figure()
        ax = fig.add_subplot( 111, projection = '3d' )
        for tree in self.__subtrees:
            if tree is not None:
                tree.show( ax )
        plt.show( block = True )

    @property
    def size( self ):
        return self.__size

    @property
    def points( self ):
        pts = []
        for tree in self.__subtrees:
            if tree is not None:
                pts.extend( tree.points )
        return np.vstack( pts ) if pts else []

class StaticKDTree:
    def __init__(self, data, leafsize ):
        self.__root = KDNode( data, 0, leafsize )

    def delete( self, q ):
        self.__root.delete( q )

    def search( self, q, k, nn, dmax, dfunc ):
        # if dfunc is None: dfunc = DynamicKDTree.__default_distance_metric
        # if nn is None: nn = np.inf * np.ones( ( k, np.size( q ) ) )
        return self.__root.search( q, k, nn, dmax, dfunc )

    def show( self, ax = None ):
        show = ax is None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot( 111, projection = '3d' )
        self.__root.show( ax )
        if show: plt.show( block = True )


    @property
    def points( self ):
        pts = self.__root.points
        return np.vstack( pts ) if pts else []

    @property
    def size( self ):
        return self.__root.size

class KDNode:
    def __init__(self, data, splitdim, leafsize):
        if len( data.shape ) == 1: data = np.expand_dims( data, 0 )
        numpts, numdim = data.shape
        if numpts >= 2.0 * leafsize:
            splitval = np.median( data[ :, splitdim ] )
            goleft = data[ :, splitdim ] < splitval

            leftpoints = data[ goleft, : ]                          # these points go left
            rightpoints = data[ np.logical_not( goleft ), : ]       # these points go right

            self.__data = None
            self.__splitdim = splitdim
            self.__splitval = splitval
            
            self.__leftchild = KDNode( leftpoints, ( splitdim + 1 ) % numdim, leafsize )
            self.__rightchild = KDNode( rightpoints, ( splitdim + 1 ) % numdim, leafsize )
        else:
            self.__data = data
            self.__splitdim = None
            self.__splitval = None
            self.__leftchild = None
            self.__rightchild = None

        self.__size = numpts

    def delete( self, q ):
        if self.__leftchild is None and self.__rightchild is None: # this is a leaf
            # if len( q.shape ) == 1: q = np.expand_dims( q, 0 )
            to_del = []
            for row in q:
                idx = np.argwhere( ( self.__data == row ).all(-1) ).reshape(-1)
                to_del.extend( idx )
            if to_del: self.__data = np.delete( self.__data, to_del, axis = 0 )
            self.__size -= len( to_del )
        else:
            goleft = q[ :, self.__splitdim ] < self.__splitval
            if np.any( goleft ): self.__leftchild.delete( q[ goleft, : ] )
            if np.any( np.logical_not( goleft ) ): self.__rightchild.delete( q[ np.logical_not( goleft ), : ] )
            
            self.__size = self.__leftchild.size + self.__rightchild.size 
            
            if not self.__leftchild.__size: self.__leftchild = None
            if not self.__rightchild.__size: self.__rightchild = None
            
            if self.__leftchild is None and self.__rightchild is not None: 
                self.__data = np.vstack( self.__rightchild.points )
                self.__rightchild = None
            elif self.__rightchild is None and self.__leftchild is not None: 
                self.__data = np.vstack( self.__leftchild.points )
                self.__leftchild = None

    def search( self, q, k, nn, dmax, dfunc ):
        if self.__leftchild is None and self.__rightchild is None:
            # this is a leaf node
            pts = np.vstack( [ nn, self.__data ] )
            dall = dfunc( q, pts, axis = 1 )
            idx = np.argsort( dall )[ :k ]
            nn = pts[ idx, : ]
            dmax = dall[ idx[ -1 ] ]
        else:
            # this is a parent node
            goleft = q[ self.__splitdim ] < self.__splitval
            if goleft:
                if q[ self.__splitdim ] - dmax < self.__splitval: 
                    nn, dmax = self.__leftchild.search( q, k, nn, dmax, dfunc )
                if q[ self.__splitdim ] + dmax >= self.__splitval:
                    nn, dmax = self.__rightchild.search( q, k, nn, dmax, dfunc )
            else:
                if q[ self.__splitdim ] + dmax >= self.__splitval:
                    nn, dmax = self.__rightchild.search( q, k, nn, dmax, dfunc )
                if q[ self.__splitdim ] - dmax < self.__splitval:
                    nn, dmax = self.__leftchild.search( q, k, nn, dmax, dfunc )
        return nn, dmax

    def show( self, ax ):
        if self.__leftchild is None and self.__rightchild is None:
            if self.__data.shape[1] != 3: pass
            else: ax.scatter( self.__data[:,0], self.__data[:,1], self.__data[:,2],
                              c = np.random.rand(3,) )
        else:
            self.__leftchild.show( ax )
            self.__rightchild.show( ax )

    @property
    def points(self):
        pts = []
        if self.__leftchild is not None: pts.extend( self.__leftchild.points )
        if self.__rightchild is not None: pts.extend( self.__rightchild.points )
        if self.__data is not None: pts.append( self.__data )
        return pts

    @property
    def size(self):
        return self.__size

if __name__ == "__main__":
    import time

    NUM_DIM = 3
    NUM_PTS = 20000
    NUM_BATCH = 100

    NUM_DEL = NUM_PTS // 2 // NUM_BATCH
    NUM_KNN = 200

    pts = np.random.rand( NUM_PTS, NUM_DIM )
    batchsize = NUM_PTS // NUM_BATCH

    t1 = time.clock()
    tree = DynamicKDTree()
    t2 = time.clock()
    print( "Creating dynamic KD Tree...", 1000 * ( t2 - t1 ), 'ms' )

    t = np.zeros( ( NUM_BATCH, 3 ) )
    for batch in range( 0, NUM_BATCH ):
        print( '\nBatch', batch+1 )
        batchidx = batch * batchsize
        batchpts = ( batch + 1 ) * pts[ batchidx:batchidx+batchsize, : ]

        t1 = time.clock()
        tree.insert( batchpts )
        t2 = time.clock()
        t[ batch, 0 ] = 1000 * ( t2 - t1 )
        print( "Iterative point addition...", t[batch, 0], 'ms, (', 
                                              t[batch,0] / batchsize, 'ms per addition )'  )

        idx = np.arange( batchsize )
        np.random.shuffle( idx )
        idx = idx[ :NUM_DEL ]
    
        t1 = time.clock()
        tree.delete( batchpts[ idx, : ] )
        t2 = time.clock()
        t[ batch, 1 ] = 100 * ( t2 - t1 )
        print( 'Recursive point deletion...', t[batch,1], 'ms, (', 
                                              t[batch,1] / NUM_DEL, 'ms per deletion )' )

        t1 = time.clock()
        for p in batchpts[ idx, : ]: knn = tree.search( p, NUM_KNN )
        t2 = time.clock()
        t[ batch, 2 ] = 1000 * ( t2 - t1 )
        print( 'Nearest neighbor search...', t[batch,2], 'ms, (',
                                             t[batch,2] / NUM_DEL, 'ms per search )' )

        print( 'KD Tree size:', tree.size )

    tree.show()