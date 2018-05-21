import sys
from os.path import dirname, abspath
sys.path.insert( 0, dirname( dirname( abspath( __file__ ) ) ) )

from mite.filters import TimeDomainFilter, JointPositionFilter, MinMaxFilter
from mite.classifiers import *
from mite.utils.Metrics import confusion_matrix

import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    TRAINING_DATA = 'train_19Apr2018_03-20-12_PM.pdata'
    WINDOW_SIZE = 10                                # in samples
    WINDOW_STEP = 2
    MODEL = 'SRC'
    
    print( 'Loading training data...' )
    data = pickle.load( open( TRAINING_DATA, 'rb' ) )
    X = []                                          # will hold features
    y = []                                          # will hold labels
    S = []                                          # will hold states

    print( 'Creating filters...' )
    td5 = TimeDomainFilter( num_features = 5 )
    # jpos = JointPositionFilter( num_quaternions = 5 )
    
    grasps = [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]
    for trial in data:                              # for each trial
        key_count = 0
        for grasp in grasps:                        # for each grasp
            print( grasp, '...', end = ' ', flush = True )
            emg = trial[ grasp ][ 'MyoArmband' ][ :, :8 ]  # grab the raw EMG data
            # quat = trial[ grasp ][ 'IMU' ]                 # grab the raw IMU data
            
            # calculating stuff
            feat = []
            # state = []
            label = []
            n_samples = emg.shape[0]
            for sample in range( WINDOW_SIZE, n_samples, WINDOW_STEP ):
                feat.append( td5.filter( emg[sample-WINDOW_SIZE:sample, :] ) )                              # calculate features
                # state.append( np.mean( jpos.filter( quat[sample-WINDOW_SIZE:sample, :] ), axis=0 )[9:] )    # average state value over feature window
                label.append( key_count )
            X.append( np.vstack( feat ) )
            # S.append( np.vstack( state ) )
            y.append( np.vstack( label ) )

            key_count += 1

    # these are now matrices
    X = np.vstack( X )
    y = np.squeeze( np.vstack( y ) )
    # S = np.vstack( S )

    print( '\nPreprocess data...', end = ' ', flush = True )
    t1 = time.clock()
    mmfilter = MinMaxFilter( X, dynamic = True )
    X = mmfilter.filter( X )
    t2 = time.clock()
    print( 1000 * ( t2 - t1 ), 'ms' )

    print('Dividing data...', end = ' ', flush = True)
    t1 = time.clock()
    Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, test_size = 0.33 )
    t2 = time.clock()
    print( 1000 * ( t2 - t1 ), 'ms' )

    print('Creating classifiers...', end= ' ', flush = True)
    t1 = time.clock()
    mdl = eval( MODEL )( Xtrain, ytrain )
    t2 = time.clock()
    print( 1000 * ( t2 - t1 ), 'ms' )

    print('Testing classifier...', end = ' ', flush = True)
    t1 = time.clock()
    yhat, yprob = mdl.predict( Xtest, prob = True )
    t2 = time.clock()
    print( 1000 * ( t2 - t1 ), 'ms (', 1000 * ( t2 - t1 ) / np.size( yhat ), 'ms per decision )' )

    print('Classifier done!')

    class_labels = [ 'RE', 'HO', 'HC', 'TR', 'WP', 'WS' ]
    cm = confusion_matrix( ytest, yhat, labels = class_labels, cmap = 'Blues' )

if __name__ == '__main__':
    main()