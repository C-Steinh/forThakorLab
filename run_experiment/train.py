import sys
sys.path.insert( 0, '/home/cynthia/forThakorLab' )

from mite.inputs import DebugDevice, MyoArmband, InertialMeasurementUnits
from mite.protocols import OfflineTrainer

import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--debug', type = int, action = 'store',
                     dest = 'debug', default = 0 )
parser.add_argument( '--trials', type = int, action = 'store',
                     dest = 'trials', default = 1 )
parser.add_argument( '--duration', type = int, action = 'store',
                     dest = 'duration', default = 3 )
args = parser.parse_args()

if args.debug:
    print( 'Creating DBG interface...' )
    dbg = DebugDevice( num_channels = 8, srate = 200.0 )
    hw_list = [ dbg ]
else: 
    # print( 'Creating IMU interface' )
    # imu = InertialMeasurementUnits( com = '/dev/ttyACM0', chan = [ 0, 1, 2, 3, 4 ], srate = 50.0 )

    print( 'Creating MYO interface...' )
    myo = MyoArmband( com = '/dev/ttyACM0', mac = 'eb:33:40:96:ce:a5', srate = 200.0 )
    #two options for mac address: one is: 'ff:f5:c9:fc:bc:17' the other 'eb:33:40:96:ce:a5'

    hw_list = [ myo ] #, imu ]

print( 'Setting up training protocol...' )
cue_list = [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]
trainer = OfflineTrainer( num_trials = args.trials, duration = args.duration, 
                          cues = cue_list, hardware = hw_list )

print( 'Press ENTER to begin...' )
input()
trainer.view()
data = trainer.run()
trainer.hide()

# plot the data
figs = []
n_cues = len( cue_list )
for i in range( 0, len( data ) ):
    figs.append( plt.figure() )
    n_cols = np.ceil( np.sqrt( n_cues ) )
    n_rows = np.floor( np.sqrt( n_cues ) )
    for j in range( 0, n_cues ):
        ax = figs[ i ].add_subplot( n_rows, n_cols, j + 1 )
        for ch in range( 0, 8 ):
            if args.debug:
                ax.plot( data[ i ][ cue_list[ j ] ][ 'Time' ],
                            data[ i ][ cue_list[ j ] ][ 'Debug' ][ :, ch ] + 2 * ( ch + 1 ) )
            else:
                ax.plot( data[ i ][ cue_list[ j ] ][ 'Time' ],
                            data[ i ][ cue_list[ j ] ][ 'MyoArmband' ][ :, ch ] + 128 + 256 * ch )
        if args.debug: print( data[ i ][ cue_list[ j ] ][ 'Debug' ].shape )
        else: print( data[ i ][ cue_list[ j ] ][ 'MyoArmband' ].shape )
        ax.set_title( cue_list[ j ] )
    figs[ i ].canvas.set_window_title( 'Trial {0}'.format( i + 1 ) )
plt.show( block = True )

# save the data
filename = 'train_' + datetime.datetime.now().strftime( "%d%b%Y_%I-%M-%S_%p" ) + '.pdata'
pkl_data = pickle.dumps( data, pickle.HIGHEST_PROTOCOL )
with open( filename, 'wb' ) as f:
    f.write( pkl_data )
    f.close()

print( 'Done...' )
