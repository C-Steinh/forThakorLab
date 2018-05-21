import sys
import datetime
import pickle

import time
import numpy as np
import threading as th

from .. import ns_sleep

__HEADER_DELIMETER__ = b'\x00\xff\x00\xff\x00\xff'

def __split_file( f, delim = __HEADER_DELIMETER__, bufsize = 1024 ):
    prev = b''
    while True:
        s = f.read( bufsize )
        if not s: break
        split = s.split( delim )
        if len( split ) > 1:
            yield prev + split[ 0 ]
            prev = split[ -1 ]
            for x in split[1:-1]: yield x
        else: prev += s
        if prev: yield prev

def read_from_file( filename ):
    pfile = []
    for s in __split_file( open( filename, 'rb' ) ): pfile.append( s )
    return pickle.loads( pfile[-1] )

def write_to_file( stream, filename ):
    output = stream.flush()
    pkl_out = pickle.dumps( output, pickle.HIGHEST_PROTOCOL )
    with open( filename, 'w' ) as f:
        f.write( stream._create_header() )
        f.write( 'Size of Data in Bytes: ' + str( sys.getsizeof( pkl_out ) ) + '\n' )
        f.close()
    with open( filename, 'ab' ) as f:
        f.write( __HEADER_DELIMETER__ )
        f.write( pkl_out )
        f.close()

class DataStream:
    def __init__(self, hardware = None):
        self.__hardware = hardware
        self.__speriod = np.inf                             # initialize sampling rate
        self.__state = { 'Time' : [] }                      # initialize empty dictionary
        for dev in self.__hardware:
            self.__state.update( { dev.name : [] } )        # add device name as key to dictionary
            if dev.speriod < self.__speriod:            
                self.__speriod = dev.speriod                # update writer's sampling period if greater than device's
        
        self.__recording = th.Event()
        self.__pause = th.Event()
        self.__thread = None

    def _create_header( self ):
        header = datetime.datetime.now().strftime( "File written on %d %b %Y at %I:%M:%S %p\n" )
        for dev in self.__hardware:
            header += type( dev ).__name__ + ' Interface named '
            header += dev.name + ',  ' 
            header += repr( dev.channelcount ) + ' Channels with Sampling Rate ' 
            header += repr( 1.0 / dev.speriod ) + ' Hz\n'
        return header

    def __record( self ):
        # t = time.clock()
        while self.__recording.is_set():
            if not self.__pause.is_set():
                t = time.clock() + self.__speriod
                self.__state[ 'Time' ].append( time.clock() )
                for dev in self.__hardware:
                    # print( dev.state[:8] )
                    self.__state[ dev.name ].append( dev.state ) #FIXME: THIS IS NOT UPDATING
                while max( t - time.clock(), 0 ): ns_sleep( 100 )

    def start_record( self ):
        if not self.__recording.is_set():
            self.__recording.set()
            self.__thread = th.Thread( target = self.__record )
            self.__thread.start()

    def stop_record( self ):
        if self.__recording.is_set():
            self.__recording.clear()
            self.__thread.join()

    def peek( self ):
        output = self.__state.copy()       # copy to keep atomicity
        output[ 'Time' ] = np.array( output[ 'Time' ] )
        for dev in self.__hardware:
            key = dev.name
            output[ key ] = np.vstack( output[ key ] )
        return output

    def flush( self ):
        self.__pause.set()
        output = self.__state.copy()       # copy to keep atomicity
        output[ 'Time' ] = np.array( output[ 'Time' ] )
        self.__state[ 'Time' ] = []
        for dev in self.__hardware:
            key = dev.name
            output[ key ] = np.vstack( output[ key ] )
            self.__state[ key ] = []
        self.__pause.clear()
        return output

if __name__ == '__main__':
    from ..inputs import DebugDevice

    dbg = DebugDevice( srate = 100.0 )
    stream = DataStream( hardware = [ dbg ] )
    
    dbg.run( display = False )
    stream.start_record()
    print( 'Start Recording...' )
    time.sleep( 3 )
    stream.stop_record()
    print( 'Stop Recording...' )
    dbg.stop()

    tmpfile = 'tmp_datafile_' + datetime.date.today().strftime( "%d%b%Y" ) + '.pdata'
    write_to_file( stream, filename = tmpfile )

    data = read_from_file( tmpfile )
    print( repr( data[ 'Debug' ].shape ) )