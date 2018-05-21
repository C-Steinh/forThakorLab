import time
import threading as th
import numpy as np

import multiprocessing as mp
import matplotlib.pyplot as plt

from .. import ns_sleep

class DebugDevice:
    """ Python implementation of a fake hardware device for debug purposes """

    def __init__(self, name = 'Debug', num_channels = 8, srate = 100.0):
        """ Constructor """
        self.__name = name
        self.__channelcount = num_channels
            
        # streaming variables
        self.__state = np.zeros(self.__channelcount)
        self.__speriod = 1.0 / srate
        self.__thread = None

        self.__stream_data = th.Event()
        self.__print_data = False

        # viewing variables
        self.__queue = mp.Queue()
        self.__viewer = None
        self.__view = mp.Event()

    def __del__(self):
        """ Destructor """
        try:
            if self.__thread.is_alive:
                self.__thread.join()
        except AttributeError: # never got to make the I/O thread
            pass
        try:
            if self.__viewer.is_alive:
                self.hide()
        except AttributeError: # no viewer exists currently
            pass

    def __read(self):
        """ Reads a single sample from the debug device """
        self.__state = np.random.rand( self.__channelcount )
        if self.__print_data: print( self.__state )
        
    def __stream(self):
        """ Streams data from the debug device at the specified sampling rate """
        t = time.clock()
        while self.__stream_data.is_set():
            t = t + self.__speriod
            self.__read()
            while max( t - time.clock(), 0 ): ns_sleep( 100 ) #  pass #time.sleep( 0.001 )

    def __plot( self ):
        pass

    @property
    def name(self):
        """ Returns the associated name for the debug device """
        return self.__name

    @property
    def state(self):
        """ Returns the current state of the debug device """
        return self.__state

    @property
    def speriod(self):
        """ Returns the sampling period of the debug device """
        return self.__speriod

    @property
    def channelcount(self):
        """ Returns the number of channels for the debug device """
        return self.__channelcount

    def run(self, display = False):
        """ Starts the acquisition thread of the debug device """
        if not self.__stream_data.is_set():
            self.__stream_data.set()
            self.__print_data = display
            self.__thread = th.Thread(target=self.__stream)
            self.__thread.start()

    def stop(self):
        '''Stops the acquisition thread of the debug device'''
        if self.__stream_data.is_set():
            self.__stream_data.clear()
            self.__thread.join()

    def view(self):
        """ Launches the GUI viewer of the debug device data """
        pass
        if not self.__view.is_set():
            self.__view.set()
            self.__viewer = mp.Process( target = self.__plot )
            self.__viewer.start()
            
    def hide(self):
        '''Closes the GUI viewer of the debug device data'''
        if self.__view.is_set():
            self.__view.clear()
            self.__viewer.join()

if __name__ == '__main__':
    import sys
    import inspect
    import argparse

    # parse commandline entries
    class_init = inspect.getargspec( DebugDevice.__init__ )
    arglist = class_init.args[1:]   # first item is always self
    defaults = class_init.defaults
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

    dbg = DebugDevice( num_channels = args.num_channels, srate = args.srate )
    dbg.run( display = True )
    dbg.view()
