import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

class DynamicKDMap:
    @staticmethod
    def __default_distance_metric( y, X, axis = 1 ):
        return np.sum( np.square( np.subtract( X, y ) ), axis = axis )

    def __init__( self, leafsize = 100 ):
        self.__leafsize = leafsize
        self.__keysize = 0
        self.__valsize = 0
        self.__submaps = [ None ]
        self.__size = 0

    def insert( self, keys, vals ):
        if len( keys.shape ) == 1: keys.resize( ( 1, keys.shape[0] ) )
        if len( vals.shape ) == 1: vals.resize( ( 1, vals.shape[0] ) )
        assert( keys.shape[0] == vals.shape[0] )

        if not self.__keysize: self.__keysize = keys.shape[1]
        if not self.__valsize: self.__valsize = vals.shape[1]

        records = np.hstack( [ keys, vals ] )
        self.__size += records.shape[0]

        for rec in records:
            idx = self.__submaps.index( None ) if None in self.__submaps else -1
            if idx == -1:
                self.__submaps.append( None )
                idx += len( self.__submaps )

            addrecs = [ rec ]
            for i in range( idx - 1, -1, -1 ):
                addrecs.append( self.__submaps[ i ].records )
                self.__submaps[ i ] = None
            addrecs = np.vstack( addrecs )
            
            self.__submaps[ idx ] = StaticKDMap( addrecs, self.__keysize, self.__valsize, self.__leafsize )      # create new submap

    def delete( self, keys ):
        if len( keys.shape ) == 1: keys.resize( ( 1, keys.shape[0] ) )
        self.__size = 0
        for m in reversed( self.__submaps ):        # start from largest map to save calls 
            if m is not None: 
                m.delete( keys )
                self.__size += m.size

    def search( self, tgtkey, k, dfunc = None ):
        assert( np.size( tgtkey ) == self.__keysize )
        if dfunc is None: dfunc = DynamicKDMap.__default_distance_metric
        
        nn = np.inf * np.ones( ( k, self.__keysize + self.__valsize ) )
        dmax = np.inf
        for m in self.__submaps:
            if m is not None: nn, dmax = m.search( tgtkey, k, nn, dmax, dfunc )
        
        noninf = ~np.isinf( nn ).any( axis = 1 )
        if noninf.any(): return nn[noninf,:self.__keysize], nn[noninf,self.__keysize:]
        else: return None, None

    def shrink( self, tgtsize ):
        if self.__size > tgtsize:
            pctdel = tgtsize / self.__size
            mask = np.concatenate( [ np.ones( int( np.ceil( ( 1.0 - pctdel ) * self.__size ) ), dtype = np.int64 ), 
                                     np.zeros( int( np.floor( pctdel * self.__size ) ), dtype = np.int64 ) ] )
            np.random.shuffle( mask )
            recs = self.records
            if mask.sum() <= 10.0 * ( self.__size - mask.sum() ): 
                self.delete( recs[ mask, :self.__keysize] )
            else:
                self.__size = 0
                self.__submaps = [ None ]
                self.insert( recs[ ~mask, :self.__keysize ], recs[ ~mask, self.__keysize: ] )

    def show( self ):
        fig = plt.figure()
        ax = fig.add_subplot( 111, projection = '3d' )
        for m in self.__submaps:
            if m is not None: m.show( ax )
        plt.show( block = True )

    @property
    def records( self ):
        recs = []
        for m in self.__submaps:
            if m is not None: recs.extend( m.records )
        return np.vstack( recs )

    @property
    def size( self ):
        return self.__size

class StaticKDMap:
    def __init__(self, data, keysize, valsize, leafsize ):
        self.__root = KDNode( data, 0, keysize, valsize, leafsize )

    def delete( self, records ):
        self.__root.delete( records )

    def search( self, key, k, nn, dmax, dfunc ):
        return self.__root.search( key, k, nn, dmax, dfunc )

    def show( self, ax ):
        self.__root.show( ax )

    @property
    def records( self ):
        recs = self.__root.records
        return np.vstack( recs ) if recs else []

    @property
    def size( self ):
        return self.__root.size

class KDNode:
    def __init__(self, data, splitdim, keysize, valsize, leafsize):
        if len( data.shape ) == 1: data.resize( ( 1, data.shape[0] )  )

        self.__keysize = keysize
        self.__valsize = valsize
        
        numrecs = data.shape[0]
        if numrecs >= 2.0 * leafsize:
            splitval = np.median( data[ :, splitdim ] )
            goleft = data[ :, splitdim ] < splitval

            leftrecs = data[ goleft, : ]        # these records go left
            rightrecs = data[ ~goleft, : ]      # these records go right

            self.__data = None
            self.__splitdim = splitdim
            self.__splitval = splitval

            self.__leftchild = KDNode( leftrecs, ( splitdim + 1 ) % keysize, keysize, valsize, leafsize )
            self.__rightchild = KDNode( rightrecs, ( splitdim + 1 ) % keysize, keysize, valsize, leafsize )
        else:
            self.__data = data
            self.__splitdim = None
            self.__splitval = None
            self.__leftchild = None
            self.__rightchild = None

        self.__size = numrecs

    def delete( self, tgtkeys ):
        if self.__leftchild is None and self.__rightchild is None: # this is a leaf
            to_del = []
            datakeys = self.__data[:, :self.__keysize]
            for key in tgtkeys:
                idx = np.argwhere( ( datakeys == key ).all(-1) ).reshape(-1)
                to_del.extend( idx )
            if to_del: self.__data = np.delete( self.__data, to_del, axis = 0 )
            self.__size -= len( to_del )
        else:
            goleft = tgtkeys[ :, self.__splitdim ] < self.__splitval
            if np.any( goleft ): self.__leftchild.delete( tgtkeys[ goleft, : ] )
            if np.any( ~goleft ): self.__rightchild.delete( tgtkeys[ ~goleft, : ] )
            
            self.__size = self.__leftchild.size + self.__rightchild.size 
            
            if not self.__leftchild.__size: self.__leftchild = None             # left child empty
            if not self.__rightchild.__size: self.__rightchild = None           # right child empty
            
            if self.__leftchild is None and self.__rightchild is not None:      # empty left child, valid right child
                self.__data = np.vstack( self.__rightchild.records )            # collapse right child onto self
                self.__rightchild = None
            elif self.__rightchild is None and self.__leftchild is not None:    # empty right child, valid left child
                self.__data = np.vstack( self.__leftchild.records )             # collapse left child onto self
                self.__leftchild = None

    def search( self, tgtkey, k, nn, dmax, dfunc ):
        if self.__leftchild is None and self.__rightchild is None:
            # this is a leaf node
            recs = np.vstack( [ nn, self.__data ] )
            dall = dfunc( tgtkey, recs[ :, :self.__keysize ], axis = 1 )
            idx = np.argsort( dall )[ :k ]
            nn = recs[ idx, : ]
            dmax = dall[ idx[ -1 ] ]
        else:
            # this is a parent node
            goleft = tgtkey[ self.__splitdim ] < self.__splitval
            if goleft:
                if tgtkey[ self.__splitdim ] - dmax < self.__splitval: 
                    nn, dmax = self.__leftchild.search( tgtkey, k, nn, dmax, dfunc )
                if tgtkey[ self.__splitdim ] + dmax >= self.__splitval:
                    nn, dmax = self.__rightchild.search( tgtkey, k, nn, dmax, dfunc )
            else:
                if tgtkey[ self.__splitdim ] + dmax >= self.__splitval:
                    nn, dmax = self.__rightchild.search( tgtkey, k, nn, dmax, dfunc )
                if tgtkey[ self.__splitdim ] - dmax < self.__splitval:
                    nn, dmax = self.__leftchild.search( tgtkey, k, nn, dmax, dfunc )
        return nn, dmax

    def show( self, ax ):
        if self.__leftchild is None and self.__rightchild is None:
            if self.__keysize != 3: pass
            else: ax.scatter( self.__data[:,0], self.__data[:,1], self.__data[:,2],
                              c = np.random.rand(3,) )
        else:
            self.__leftchild.show( ax )
            self.__rightchild.show( ax )

    @property
    def records(self):
        recs = []
        if self.__leftchild is not None: recs.extend( self.__leftchild.records )
        if self.__rightchild is not None: recs.extend( self.__rightchild.records )
        if self.__data is not None: recs.append( self.__data )
        return recs

    @property
    def size(self):
        return self.__size

if __name__ == "__main__":
    import time

    KEY_DIM = 3
    VAL_DIM = 40
    NUM_REC = 10000

    NUM_BATCH = 100

    NUM_DEL = NUM_REC // 2 // NUM_BATCH
    NUM_KNN = 200

    keys = np.random.rand( NUM_REC, KEY_DIM )
    vals = np.random.rand( NUM_REC, VAL_DIM )
    batchsize = NUM_REC // NUM_BATCH

    t1 = time.clock()
    kdmap = DynamicKDMap( leafsize = 100 )
    t2 = time.clock()
    print( "Creating dynamic KD Map...", 1000 * ( t2 - t1 ), 'ms' )

    t = np.zeros( ( NUM_BATCH, 3 ) )
    sizes = np.zeros( NUM_BATCH, dtype = np.int64 )
    for batch in range( 0, NUM_BATCH ):
        print( '\nBatch', batch+1 )
        batchidx = batch * batchsize
        batchkey = ( batch + 1 ) * keys[ batchidx:batchidx+batchsize, : ]
        batchval = ( batch + 1 ) * vals[ batchidx:batchidx+batchsize, : ]

        t1 = time.clock()
        kdmap.insert( batchkey, batchval )
        t2 = time.clock()
        t[ batch, 0 ] = 1000 * ( t2 - t1 )
        print( "Iterative point addition...", t[batch, 0], 'ms, (', 
                                              t[batch,0] / batchsize, 'ms per addition )'  )

        idx = np.arange( batchsize )
        np.random.shuffle( idx )
        idx = idx[ :NUM_DEL ]
    
        t1 = time.clock()
        kdmap.delete( batchkey[ idx, : ] )
        t2 = time.clock()
        t[ batch, 1 ] = 100 * ( t2 - t1 )
        print( 'Recursive point deletion...', t[batch,1], 'ms, (', 
                                              t[batch,1] / NUM_DEL, 'ms per deletion )' )

        t1 = time.clock()
        for p in batchkey[ idx, : ]: knn_key, knn_val = kdmap.search( p, NUM_KNN )
        t2 = time.clock()
        t[ batch, 2 ] = 1000 * ( t2 - t1 )
        print( 'Nearest neighbor search...', t[batch,2], 'ms, (',
                                             t[batch,2] / NUM_DEL, 'ms per search )' )

        sizes[ batch ] = kdmap.size
        print( 'KD Tree size:', sizes[ batch ] )
    
    fig = plt.figure()
    ax = fig.add_subplot( 131 )
    ax.semilogy( sizes, t[:,0] / batchsize, c = 'black', marker = 'o' )
    ax.set_title( 'Insertion Times' )
    ax.set_xlabel( 'Map Size (#records)' )
    ax.set_ylabel( 'Operation Time (ms)' )
    ax.set_ylim( 0.001, 1 )

    ax = fig.add_subplot( 132 )
    ax.semilogy( sizes, t[:,1] / NUM_DEL, c = 'black', marker = 'o' )
    ax.set_title( 'Deletion Times' )
    ax.set_xlabel( 'Map Size (#records)' )
    ax.set_ylabel( 'Operation Time (ms)' )
    ax.set_ylim( 0.001, 1 )

    ax = fig.add_subplot( 133 )
    ax.semilogy( sizes, t[:,2] / NUM_DEL, c = 'black', marker = 'o' )
    ax.set_title( 'Query Times (knn = ' + str( NUM_KNN ) + ')' )
    ax.set_xlabel( 'Map Size (#records)' )
    ax.set_ylabel( 'Operation Time (ms)' )
    ax.set_ylim( 0.1, 100 )

    plt.show( block = True )
    # kdmap.show()

    # t1 = time.clock()
    # kdmap.shrink( kdmap.size / 2 )
    # t2 = time.clock()
    # print()
    # print( 1000 * ( t2 - t1 ), 'ms for 50% shrinkage' )