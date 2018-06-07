import sys
import os
from os.path import dirname, abspath
sys.path.insert( 0, dirname( dirname( abspath( __file__ ) ) ) )

from mite.filters import TimeDomainFilter, JointPositionFilter, MinMaxFilter
from mite.classifiers import *
from mite.utils.Metrics import confusion_matrix

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def process_data(grasps, data):

    WINDOW_SIZE = 10                                # in samples
    WINDOW_STEP = 2
    MODEL = 'SRC'
    
    X = []                                          # will hold features
    y = []                                          # will hold labels
    S = []                                          # will hold states

    print( 'Creating filters...' )
    td5 = TimeDomainFilter( num_features = 5 )
    jpos = JointPositionFilter( num_quaternions = 5 )

    for trial in data:                              # for each trial
        key_count = 0
        for grasp in grasps:                        # for each grasp
            print( grasp, '...') #, flush = True )
            emg = trial[ grasp ][ 'MyoArmband' ][ :, :8 ]  # grab the raw EMG data
           # plot(emg)
           # print(emg.shape)

           # quat = trial[ grasp ][ 'IMU' ]                 # grab the raw IMU data
            
            # calculating stuff
            feat = []
            state = []
            label = []
            n_samples = emg.shape[0]

            for sample in range( WINDOW_SIZE, n_samples, WINDOW_STEP ):
                feat.append( td5.filter( emg[sample-WINDOW_SIZE:sample, :] ) )                              # calculate features
               # state.append( np.mean( jpos.filter( quat[sample-WINDOW_SIZE:sample, :] ), axis=0 )[9:] )    # average state value over feature window
                label.append( key_count )
            X.append( np.vstack( feat ) )
            #S.append( np.vstack( state ) )
            y.append( np.vstack( label ) )

            key_count += 1
            print(key_count)

    # these are now matrices
   # print(X)
    X = np.vstack( X )
    y = np.squeeze( np.vstack( y ) )
   # S = np.vstack( S )
    print(y)

    print( '\nPreprocess data...') #, end = ' ', flush = True )
#    t1 = time.clock()
    mmfilter = MinMaxFilter( X, dynamic = True )
    X = mmfilter.filter( X )
#    t2 = time.clock()
#    print( 1000 * ( t2 - t1 ), 'ms' )

    print('Dividing data...') #, end = ' ', flush = True)
#    t1 = time.clock()
    Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, test_size = 0.33 )
#    t2 = time.clock()
#    print( 1000 * ( t2 - t1 ), 'ms' )

    print('Creating classifiers...') #, end= ' ', flush = True)
#    t1 = time.clock()
    mdl = eval( MODEL )( Xtrain, ytrain )
#    t2 = time.clock()
#    print( 1000 * ( t2 - t1 ), 'ms' )

    print('Testing classifier...') #, end = ' ', flush = True)
#    t1 = time.clock()
    yhat, yprob = mdl.predict( Xtest, prob = True )
#    t2 = time.clock()
#    print( 1000 * ( t2 - t1 ), 'ms (', 1000 * ( t2 - t1 ) / np.size( yhat ), 'ms per decision )' )

    print('Classifier done!')

    class_labels = [ 'RE', 'HO', 'HC', 'TR', 'WP', 'WS' ]
    cm = confusion_matrix( ytest, yhat, labels = class_labels, cmap = 'Blues' )

def shift_left(orig_lst, n):
    """Shifts the lst over by n indices

    >>> lst = [1, 2, 3, 4, 5]
    >>> shift_left(lst, 2)
    >>> lst
    [3, 4, 5, 1, 2]
    """
    lst= orig_lst
    if n < 0:
        raise ValueError('n must be a positive integer')
    if n > 0:
        if n % 1 == 0:
            lst.insert(0, lst.pop(-1))  # shift one place
            shift_left(lst, n-1)  # repeat
        else:
            int_n = float(round(n))
            dec_n = float(n- round(n))

            lst.insert(0, lst.pop(-1))  # shift one place
            shift_left(lst, int_n-1)  # repeat
            list = [float(item) +dec_n for item in lst]
            
    return lst
        
def openNplot(TRAINING_DATA):
    """ opens the file names above and makes readable by grasp type"""
    print( 'Loading training data...' )
    data = pickle.load( open( TRAINING_DATA, 'rb' ) )
      
    grasps = [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]
    #current data only contains one 'trial' with several grasps in it
    #process_data(grasps,data)
    for trial in data:                              # for each trial
        return trial

#####################################################################
########################The main function############################

def main():
    relv_files = []
    #find all files in the directory
    dirFiles = os.listdir('.')
    for files in dirFiles:
        if 'train' in files:
            relv_files.append(files)
    relv_files = (sorted(relv_files))

    # for one file - load in CANONICAL arrangement for analysis
    
    CANON_DATA = relv_files[0]
    CANON_trial = openNplot(CANON_DATA)
    grasps = [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]
    grasp = grasps[1]
 
    #shifts the original arrangement by the amount of the shifts in the experiment
   
    shift_amounts =[0, 1, 2, 3, 4, 5, 6, 7, 8]#also 8.5 but will use when interpolating

    for shifts in range(1,len(relv_files)-1):
        orig_arrange = [0, 1, 2, 3, 4, 5, 6, 7]
    
        shift_DATA = relv_files[shifts]
        shift_trial = openNplot(shift_DATA)
        
        shifted_orig = shift_left(orig_arrange, shift_amounts[shifts])
        print(shifted_orig)
       
        #plot shift one right actual and predicted

        for grasp in grasps:
       # plt.ioff()
            fig, axes = plt.subplots(nrows =4, ncols=2, figsize = (7,7))

            for numElec in range(0,8):
                ax = plt.subplot(4,2,numElec+1)
                plt.plot(CANON_trial[grasp]['MyoArmband'][:,shifted_orig[numElec]])
                plt.plot(shift_trial[grasp]['MyoArmband'][:,numElec])
                ax.set_title(''.join(['Elec: ',str(numElec), ' Shift: ', str(shift_amounts[shifts])]))
            
                # print(numElec)
                fig.suptitle(grasp)      
                #  plt.show()
                plt.savefig(''.join(['shift_',str(shifts),grasp]))
        plt.close('all')

if __name__ == '__main__':
    main()
