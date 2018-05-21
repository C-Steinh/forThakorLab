import pygatt
import struct
import time
import threading as th
import numpy as np

import multiprocessing as mp
import matplotlib.pyplot as plt

from .. import ns_sleep
from ..utils import Quaternion as quat

class MyoArmband:
    """ Python implementation of a Myo Armband driver from Thalmic Labs  """
    MYO_CMD_CHARACTERISTIC       = 'd5060401-a904-deb9-4748-2c7f4a124842'
    MYO_IMU_DATA_CHARACTERISTIC  = 'd5060402-a904-deb9-4748-2c7f4a124842'
    MYO_EMG_DATA0_CHARACTERISTIC = 'd5060105-a904-deb9-4748-2c7f4a124842'
    MYO_EMG_DATA1_CHARACTERISTIC = 'd5060205-a904-deb9-4748-2c7f4a124842'
    MYO_EMG_DATA2_CHARACTERISTIC = 'd5060305-a904-deb9-4748-2c7f4a124842'
    MYO_EMG_DATA3_CHARACTERISTIC = 'd5060405-a904-deb9-4748-2c7f4a124842'
    def __init__(self, name = 'MyoArmband', com = '/dev/ttyACM0', mac = 'd2:8c:41:0e:05:33', srate = 200.0 ):
        """ Constructor """
        
    
        self.__name = name
        self.__btle = pygatt.BGAPIBackend( serial_port = com )

        try:
            self.__btle.start()
            self.__myo = self.__btle.connect( mac )
            self.__emg_value = None
            self.__emg_time = None

            self.__myo.char_write(MyoArmband.MYO_CMD_CHARACTERISTIC, b'\x01\x03\x00\x00\x00')   # deregister all streaming
            self.__myo.char_write(MyoArmband.MYO_CMD_CHARACTERISTIC, b'\x0a\x01\x02')           # lock myo
            self.__myo.char_write(MyoArmband.MYO_CMD_CHARACTERISTIC, b'\x09\x01\x01')           # don't sleep
            self.__myo.char_write(MyoArmband.MYO_CMD_CHARACTERISTIC, b'\x01\x03\x02\x01\x00')   # stream filtered emg and imu
            
            self.__myo.subscribe(MyoArmband.MYO_IMU_DATA_CHARACTERISTIC,  callback = None)
            self.__myo.subscribe(MyoArmband.MYO_EMG_DATA0_CHARACTERISTIC, callback = self.__emg_handler)
            self.__myo.subscribe(MyoArmband.MYO_EMG_DATA1_CHARACTERISTIC, callback = self.__emg_handler)
            self.__myo.subscribe(MyoArmband.MYO_EMG_DATA2_CHARACTERISTIC, callback = self.__emg_handler)
            self.__myo.subscribe(MyoArmband.MYO_EMG_DATA3_CHARACTERISTIC, callback = self.__emg_handler)
        except:
            raise RuntimeError('Could not connect to myo armband!')

        # set streaming parameters
        self.__channelcount = 12
        self.__state = np.zeros(self.__channelcount)
        self.__speriod = 1.0 / srate
        self.__thread = None

        self.__stream_data = th.Event()
        self.__print_data = False

        # viewing variables
        self.__queue = mp.Queue()
        self.__viewer = None
        self.__view_data = mp.Event()

    def __del__(self):
        """ Destructor """
        try:
            if self.__btle is not None:
                self.__btle.stop()
                self.__myo.disconnect()
        except AttributeError: # never made the BTLE
            pass
        try:
            if self.__thread.is_alive:
                self.__thread.stop()
        except AttributeError: # never got to make the I/O thread
            pass
        try:
            if self.__viewer.is_alive:
                self.hide()
        except AttributeError: # no viewer exists currently
            pass

    def __emg_handler(self, handle, value):
        self.__emg_value = np.array( struct.unpack( 16*'b', value ) )
        self.__emg_time = time.clock()

    def __read(self):
        """ Reads a single sample from the MyoArmband """
        qbytes = self.__myo.char_read( MyoArmband.MYO_IMU_DATA_CHARACTERISTIC )
        self.__state[8:] = np.array( struct.unpack(10*'h', qbytes)[:4] ) / 16384.0
        
        if self.__emg_value is not None:
            emg_idx = 8 * ( time.clock() - self.__emg_time > 0.005 )
            self.__state[:8] = self.__emg_value[emg_idx:emg_idx+8]

        if self.__print_data: print( self.__state )
        if self.__view_data.is_set(): self.__queue.put( self.__state )
            
    def __stream(self):
        """ Streams data from the MyoArmband at the specified sampling rate """
        t = time.clock()
        while self.__stream_data.is_set():
            t = t + self.__speriod
            self.__read()
            while max( t - time.clock(), 0 ): ns_sleep( 100 )

    def __plot(self):
        gui = plt.figure()
        gui.canvas.set_window_title( self.__name )
        
        # orientation plots
        orientations = []
        orient_colors = 'rgb'
        orient_titles = [ 'Roll', 'Pitch', 'Yaw' ]
        for i in range( 0, 3 ):
            ax = gui.add_subplot( 2, 3, i + 1, 
                 projection = 'polar', aspect = 'equal' )
            ax.plot( np.linspace( 0, 2*np.pi, 100 ), 
                     np.ones( 100 ), color = orient_colors[ i ],
                     linewidth = 2.0 )
            orientations.append( ax.plot( np.zeros( 2 ), np.linspace( 0, 1, 2 ), 
                                         color = orient_colors[ i ], linewidth = 2.0  ) )
            ax.set_rticks( [] )
            ax.set_rmax( 1 )
            ax.set_title( orient_titles[ i ] )
            ax.grid( True )

        # line plot
        emg_plots = []
        emg_offsets = np.array( [ 128, 384, 640, 896, 1152, 1408, 1664, 1920 ] )
        ax = gui.add_subplot( 2, 1, 2 )
        num_emg_samps = int( 5 * np.round( 1.0 / self.__speriod ) )
        for i in range( 0, 8 ):
            xdata = np.linspace( 0, 1, num_emg_samps )
            ydata = emg_offsets[ i ] * np.ones( num_emg_samps )
            emg_plots.append( ax.plot( xdata, ydata ) )
        ax.set_ylim( 0, 2048 )
        ax.set_xlim( 0, 1 )
        ax.set_yticks( emg_offsets.tolist() )
        ax.set_yticklabels( [ 'EMG01', 'EMG02', 'EMG03', 'EMG04', 'EMG05', 'EMG06', 'EMG07', 'EMG08' ] )
        ax.set_xticks( [] )  

        plt.show( block = False )
        while self.__view_data.is_set():
            try:
                data = []
                while self.__queue.qsize() > 0: data.append( self.__queue.get() )
                if data:
                    # concate to get a block of data
                    data = np.vstack( data )

                    # update orientation data
                    angles = quat.to_euler( data[-1, 8:] )
                    for i in range( 0, 3 ):
                        tdata = np.ones( 2 ) * angles[ i ]
                        rdata = np.linspace( 0, 1, 2 )
                        orientations[ i ][ 0 ].set_data( tdata, rdata )
                
                    # update electrophysiological data
                    for i in range( 0, 8):
                        ydata = emg_plots[ i ][ 0 ].get_ydata()
                        ydata = np.append( ydata, data[ :, i ] + emg_offsets[ i ] )
                        ydata = ydata[-num_emg_samps:]
                        emg_plots[ i ][ 0 ].set_ydata( ydata )
                plt.pause( 0.005 )
            except: self.__view_data.clear()
        plt.close( gui )

    @property
    def name(self):
        """ Returns the associated name of the MyoArmband """
        return self.__name

    @property
    def state(self):
        """ Returns the current state of the MyoArmband """
        return self.__state.copy()

    @property
    def speriod(self):
        """ Returns the sampling period of the MyoArmband """
        return self.__speriod

    @property
    def channelcount(self):
        """ Returns the number of channels for the MyoArmband """
        return self.__channelcount

    def run(self, display = False):
        """ Starts the acquisition thread of the MyoArmband """
        if not self.__stream_data.is_set():
            self.__stream_data.set()
            self.__print_data = display
            self.__thread = th.Thread(target=self.__stream)
            self.__thread.start()

    def stop(self):
        '''Stops the acquisition thread of the IMUs'''
        if self.__stream_data.is_set():
            self.__stream_data.clear()
            self.__thread.join()

    def view(self):
        """ Launches the GUI viewer of the IMU data """
        if not self.__view_data.is_set():
            self.__view_data.set()
            self.__viewer = mp.Process( target = self.__plot )
            self.__viewer.start()
            
    def hide(self):
        '''Closes the GUI viewer of the IMU data'''
        if self.__view_data.is_set():
            self.__view_data.clear()
            self.__viewer.join()

if __name__ == '__main__':
    import sys
    import inspect
    import argparse

    # parse commandline entries
    class_init = inspect.getargspec( MyoArmband.__init__ )
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

    # create interface
    myo = MyoArmband( name = args.name, com = args.com, mac = args.mac, srate = args.srate )
    myo.run( display = False )
    myo.view()
