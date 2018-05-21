import serial
import struct
import time
import threading as th
import numpy as np
import numpy.matlib

import multiprocessing as mp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .. import ns_sleep
from ..utils import Quaternion as quat

class InertialMeasurementUnits:
    """ Python implementation of the inertial measurement unit net """

    def __init__(self, name = 'IMU', com = '/dev/ttyACM0', baud = 115200, chan = [ 0, 1, 2, 3, 4 ], srate = 50.0):
        """ Constructor """
        self.__MAX_INIT_RETRIES = 10
        self.__name = name
        self.__channels = chan
        self.__ser = serial.Serial(com, baud, timeout = 0.5)
        if self.__set_channels():
            self.__channelcount = len( chan )
            if self.__query_init():
                # streaming variables
                self.__state = np.squeeze( np.matlib.repmat( np.array( [ 1, 0, 0, 0 ], dtype = np.float ), 1, self.__channelcount ) ) # np.zeros(4 * self.__channelcount)
                self.__calibrate = np.squeeze( np.matlib.repmat( np.array( [ 1, 0, 0, 0 ], dtype = np.float ), 1, self.__channelcount ) )
                self.__speriod = 1.0 / srate
                self.__thread = None

                self.__stream_data = th.Event()
                self.__print_data = False

                # viewing variables
                self.__queue = mp.Queue()
                self.__viewer = None
                self.__view = mp.Event()
            else:
                raise RuntimeError("IMU was not initialized correctly")
        else:
            raise ValueError("IMU channels were not set correctly", 'chan')

    def __del__(self):
        """ Destructor """
        try:
            if self.__ser.is_open:
                self.__ser.close()
        except AttributeError: # never made the serial device
            pass
        try:
            if self.__thread.is_alive:
                self.stop()
        except AttributeError: # never got to make the I/O thread
            pass
        try:
            if self.__viewer.is_alive:
                self.hide()
        except AttributeError: # no viewer exists currently
            pass

    def __set_channels(self):
        """ Sets active channels for the dongle
            Returns true, false
        """
        mask = b'\x80\x40\x20\x10\x08\x04\x02\x01'
        cmd = bytearray(b'\x63\x00')
        for i in self.__channels:
            cmd[1] = cmd[1] | mask[i]
        self.__ser.write(cmd)
        ch = self.__ser.read(1)
        if len( ch ):
            ch = struct.unpack('B', ch)[0]
            return (ch == len(self.__channels))
        else: return False

    def __query_init(self):
        """ Checks for proper initialization:
            Returns true, false
        """
        for attempt in range(0, self.__MAX_INIT_RETRIES):
            self.__ser.write(b'\x69')
            init = self.__ser.read(1)
            init = struct.unpack('B', init)[0]
            if init == 121: return True
            time.sleep( 0.2 ) # wait a second
        return False
    
    def __chksum(self, b):
        return ( ( sum(bytearray(b[:-1])) % 256 ) == b[-1] )

    def __read(self):
        """ Reads a single sample from the IMUs """
        self.__ser.write(b'\x77')
        sample = self.__ser.read(16*self.__channelcount+1)
        if self.__chksum(sample):
            data = np.array(struct.unpack(4*self.__channelcount*'f', sample[0:-1]))
            for i in range( 0, self.__channelcount):
                idx1 = i * 4
                idx2 = idx1 + 4
                self.__state[idx1:idx2] = quat.relative(self.__calibrate[idx1:idx2], data[idx1:idx2])
            if self.__print_data: print( self.__state )
            if self.__view.is_set(): self.__queue.put( self.__state )
        else:
            self.__ser.flushInput()
    
    def __stream(self):
        """ Streams data from the IMUs at the specified sampling rate """
        t = time.clock()
        while self.__stream_data.is_set():
            t = t + self.__speriod
            self.__read()
            while max( t - time.clock(), 0 ): ns_sleep( 100 ) #pass

    def __plot( self ):
        ''''Update the IMU plots'''
        # initialization
        cube_x = np.array( [ [ 0, 1, 1, 0, 0, 0 ], [ 1, 1, 0, 0, 1, 1 ],
                             [ 1, 1, 0, 0, 1, 1 ], [ 0, 1, 1, 0, 0, 0 ] ] ) - 0.5
        cube_y = np.array( [ [ 0, 0, 1, 1, 0, 0 ], [ 0, 1, 1, 0, 0, 0 ],
                             [ 0, 1, 1, 0, 1, 1 ], [ 0, 0, 1, 1, 1, 1 ] ] ) - 0.5
        cube_z = np.array( [ [ 0, 0, 0, 0, 0, 1 ], [ 0, 0, 0, 0, 0, 1 ],
                             [ 1, 1, 1, 1, 0, 1 ], [ 1, 1, 1, 1, 0, 1 ] ] ) - 0.5
        cube_colors = 'rgbycm'
                
        n_rows = np.floor( np.sqrt( self.__channelcount ) )
        n_cols = np.ceil( np.sqrt( self.__channelcount ) )
                
        gui = plt.figure()
        gui.canvas.set_window_title( self.__name )
        
        polygons = []
        for i in range( 0, self.__channelcount ):
            polygons.append( [] )
            ax = gui.add_subplot( n_rows, n_cols, i + 1,
                                  projection = '3d', aspect = 'equal' )
            for side in range( 0, 6 ):
                vtx = np.array( [ cube_x[:, side],
                                  cube_y[:, side],
                                  cube_z[:, side] ] )
                poly = plt3d.art3d.Poly3DCollection( [ np.transpose( vtx ) ] )
                poly.set_color( cube_colors[ side ] )
                poly.set_edgecolor( 'k' )
                polygons[ i ].append( poly )
                
                ax.add_collection3d( polygons[ i ][ side ] )
            ax.set_xlim( ( -1, 1 ) )
            ax.set_ylim( ( -1, 1 ) )
            ax.set_zlim( ( -1, 1 ) )
            ax.set_title( 'IMU #' + repr( self.__channels[ i ] + 1 ) )
            ax.axis( 'off' )

        # stream
        plt.show( block = False )
        xnew = np.zeros( ( 4, 6 ) )
        ynew = np.zeros( ( 4, 6 ) )
        znew = np.zeros( ( 4, 6 ) )
        while self.__view.is_set():
            try:
                data = None
                while self.__queue.qsize() > 0: data = self.__queue.get()
                if data is not None:
                    for dev in range(0, self.__channelcount):
                        idx1 = dev * 4
                        idx2 = idx1 + 4
                        q = data[idx1:idx2]
                        for j in range( 0, 6 ):
                            for i in range( 0, 4 ):
                                p = np.array( [ cube_x[ i, j ], 
                                                cube_y[ i, j ],
                                                cube_z[ i, j ] ] )
                                pr = quat.rotate( q, p )
                                xnew[ i, j ] = pr[ 0 ]
                                ynew[ i, j ] = pr[ 1 ]
                                znew[ i, j ] = pr[ 2 ]
                            vtx = np.array( [ xnew[:, j], ynew[:, j], znew[:, j] ] )
                            polygons[ dev ][ j ].set_verts( [ np.transpose( vtx ) ] )
                plt.pause( 0.05 )
            except: self.__view.clear()
        plt.close( gui )

    @property
    def name(self):
        """ Returns the associated name for the IMUs """
        return self.__name

    @property
    def state(self):
        """ Returns the current state of the IMUs """
        return self.__state.copy()

    @property
    def speriod(self):
        """ Returns the sampling period of the IMUs """
        return self.__speriod

    @property
    def channelcount(self):
        """ Returns the number of channels for the IMUs """
        return self.__channelcount

    def set_calibrate(self, calibration_count = 100 ):
        """ Return calibration quaternions from the IMUs """
        Q = []
        for _ in range( 0, calibration_count ):
            self.__ser.write( b'\x77' )
            sample = self.__ser.read(16*self.__channelcount+1)
            if self.__chksum( sample ): 
                Q.append( np.array( struct.unpack( 4 * self.__channelcount*'f', sample[:-1] ) ) )
                ns_sleep( 1e4 )     # wait for 10 ms
        if len( Q ):
            Q = np.vstack( Q ).T # quaternions x samples
            for i in range( 0, self.__channelcount ):
                qidx = 4 * i
                self.__calibrate[qidx:qidx+4] = quat.average( Q[qidx:qidx+4, :] )
            return True
        else: return False

    def clear_calibrate(self):
        self.__calibrate = np.squeeze( np.matlib.repmat( np.array( [ 1, 0, 0, 0 ] ), 1, self.__channelcount ) )

    def get_calibrate(self):
        return self.__calibrate.copy()

    def run(self, display = False):
        """ Starts the acquisition thread of the IMUs """
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
        if not self.__view.is_set():
            self.__view.set()
            self.__viewer = mp.Process( target = self.__plot )
            self.__viewer.start()
            

    def hide(self):
        '''Closes the GUI viewer of the IMU data'''
        if self.__view.is_set():
            self.__view.clear()
            self.__viewer.join()

if __name__ == '__main__':
    import sys
    import inspect
    import argparse

    # parse commandline entries
    class_init = inspect.getargspec( InertialMeasurementUnits.__init__ )
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

    imu = InertialMeasurementUnits( name = args.name, com = args.com, baud = args.baud, chan = args.chan, srate = args.srate )
    imu.run( display = True )
    imu.view()
