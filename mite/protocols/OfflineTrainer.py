import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time
import random
import os.path

from .. import ns_sleep
from ..data import DataStream

class OfflineTrainer:
    def __init__( self, num_trials = 3, duration = 3, cues = [ 'rest', 'open', 'power', 'pronate', 'supinate', 'tripod' ], hardware = [] ):
        self.__num_trials = num_trials
        self.__duration = duration

        if cues:
            self.__cues = cues
            self.__cue_images = {}
            for cue in self.__cues:
                cue_path = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'cues/' )
                img = mpimg.imread( cue_path + cue + '.jpg' )
                self.__cue_images.update( { cue : img } )
        else: raise RuntimeError( 'No cues specified!' )

        if hardware:
            self.__hardware = hardware 
            self.__stream = DataStream( hardware )
        else: raise RuntimeError( 'No hardware connected!' )

    def view( self ):
        for hw in self.__hardware: hw.view()
    
    def hide( self ):
        for hw in self.__hardware: hw.hide()

    def run( self ):
        data = []
        cue_fig = plt.figure()
        plt.ion()
        for hw in self.__hardware: hw.run( display = False )
        for trial in range( 0, self.__num_trials ):
            random.shuffle( self.__cues )
            data.append( {} )                                       # append an empty dictionary
            for cue in self.__cues:                                 # randomized cues
                plt.imshow( self.__cue_images[ cue ] )              # show cue
                plt.axis( 'off' )
                plt.show( block = False )
                plt.pause( 1.0 )                                    # wait
                self.__stream.start_record()                        # start recording
                
                # ns_sleep( self.__duration * 1e9 )
                t = time.clock() + self.__duration
                while max( t - time.clock(), 0 ): ns_sleep( 100 )

                self.__stream.stop_record()                         # stop recording
                data[ trial ].update( { cue : self.__stream.flush() } )
        plt.close( cue_fig )
        for hw in self.__hardware: hw.stop()
        return data

if __name__ == '__main__':
    import sys
    import inspect
    import argparse
    from ..inputs.DebugDevice import *

    # # parse commandline entries
    class_init = inspect.getargspec( OfflineTrainer.__init__ )
    arglist = class_init.args[1:-1]   # first item is always self
    defaults = class_init.defaults[:-1]
    parser = argparse.ArgumentParser()
    for arg in range( 0, len( arglist ) ):
        try: tgt_type = type( defaults[ arg ][ 0 ] )
        except: tgt_type = type( defaults[ arg ] )
        parser.add_argument( '--' + arglist[ arg ], 
                             type = tgt_type, nargs = '+',
                             action = 'store', dest = arglist[ arg ],
                             default = defaults[ arg ] )
    args = parser.parse_args()
    for arg in range( 0, len( arglist ) ):
        attr = getattr( args, arglist[ arg ] )
        if isinstance( attr, list ) and not isinstance( defaults[ arg ], list ):
            setattr( args, arglist[ arg ], attr[ 0 ]  )
    
    # interface
    dbg = DebugDevice( srate = 100.0 )

    trainer = OfflineTrainer( num_trials = args.num_trials, duration = args.duration, cues = args.cues, hardware = [ dbg ] )
    data = trainer.run()