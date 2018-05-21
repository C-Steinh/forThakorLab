import bluetooth as bt

class Bebionic3:
    def __init__( self, mac = 'ec:fe:7e:18:3f:47' ):
        self.__bt = bt.BluetoothSocket( bt.RFCOMM )
        self.__bt.connect( ( mac, 1 ) )
        
        self.__init_bt()
        self.__bt.send( b'\xff\x02\x9e\x9f' )   # stop movement command
        self.__bt.send( b'\xff\x02\x9b\x9c' )   # clear movement command
        self.__move_dict = { 'tripod'       : b'\xff\x03\x9c\x09\xa7',
                             'power'        : b'\xff\x03\x9c\x11\xaf',
                             'pinch_open'   : b'\xff\x03\x9c\x0e\xac',
                             'active_index' : b'\xff\x03\x9c\x12\xb0',
                             'pinch_closed' : b'\xff\x03\x9c\x10\xae',
                             'key_lateral'  : b'\xff\x03\x9c\x13\xb1',
                             'index_point'  : b'\xff\x03\x9c\x14\xb2',
                             'mouse'        : b'\xff\x03\x9c\x15\xb3',
                             'column'       : b'\xff\x03\x9c\x16\xb4',
                             'relaxed'      : b'\xff\x03\x9c\x17\xb5',
                             'rest'         : b'\xff\x03\x9c\x00\x9e',
                             'open'         : b'\xff\x03\x9c\x01\x9f',
                             'pronate'      : b'\xff\x03\x9c\x24\xc2',
                             'supinate'     : b'\xff\x03\x9c\x23\xc1' }
        self.__last_move = None

    def __del__( self ):
        try:
            self.__bt.send( b'\xff\x02\x9e\x9f' )   # stop movement command
            self.__bt.send( b'\xff\x02\x9b\x9c' )   # clear movement command
            self.__bt.close()                       # close communication
        except AttributeError:
            # did not open the bluetooth communication
            pass

    def __init_bt( self ):
        self.__bt.send( b'\xff\x06\x80\x00\x00\x00\x00\x85' )                           # connect
        self.__bt.send( b'\xff\x03\x00\x82' )                                           # ack
        self.__bt.send( b'\xff\x0c\x93\x00\xff\x0c\x00\x00\x00\x0a\x0f\x64\x01\x27' )   # rest [nogrip]
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x09\xff\x0c\x00\x00\x00\x00\x0f\x64\x01\x2a' )   # standard tripod
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x0e\xff\x0c\x00\x00\x00\x02\x0f\x64\x01\x2d' )   # thumb precision open
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x10\xff\x0c\x00\x00\x00\x04\x0f\x64\x01\x31' )   # thumb precision closed
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x11\xff\x0c\x00\x00\x00\x01\x0f\x64\x01\x2f' )   # power
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x12\xff\x0c\x00\x00\x00\x03\x0f\x64\x01\x32' )   # active index
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x13\xff\x0c\x00\x00\x00\x05\x0f\x64\x01\x35' )   # key lateral
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x14\xff\x0c\x00\x00\x00\x06\x0f\x64\x01\x37' )   # finger index point
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x15\xff\x0c\x00\x00\x00\x07\x0f\x64\x01\x39' )   # mouse
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x16\xff\x0c\x00\x00\x00\x08\x0f\x64\x01\x3b' )   # column
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x17\xff\x0c\x00\x00\x00\x09\x0f\x64\x01\x3d' )   # relaxed
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x24\xff\x09\x00\x00\x00\x02\x0f\x64\x01\x40' )   # pronate
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        self.__bt.send( b'\xff\x0c\x93\x23\xff\x09\x00\x00\x00\x01\x0f\x64\x01\x3e' )   # supinate
        self.__bt.send( b'\xff\x03\x00\x93\x95' )                                       # ack
        
    def __send_movement_command( self, move ):
        if move in self.__move_dict:
            self.__bt.send( self.__move_dict[ move ] )
            self.__bt.send( b'\xff\x03\x00\x9c\x9e' )       # ack
        else:
            raise RuntimeError( 'Invalid movement class for the Bebionic3: ', move )

    def publish( self, move ):
        if move is not self.__last_move:
            self.__send_movement_command( 'rest' )
            self.__send_movement_command( move )
            self.__send_movement_command( move )    # must be sent twice
            self.__last_move = move

if __name__ == '__main__':
    bb3 = Bebionic3( mac = 'ec:fe:7e:18:3f:47' )
    moves = [ 'tripod', 'power', 'pinch_open', 'active_index', 'pinch_closed',
              'key_lateral', 'index_point', 'mouse', 'column', 'relaxed',
              'rest', 'open', 'pronate', 'supinate' ]
    print( '------------ Movement Commands ------------' )
    print( '| 00  -----  STANDARD TRIPOD CLOSED       |' )
    print( '| 01  -----  POWER                        |' )
    print( '| 02  -----  THUMB PRECISION OPEN         |' )
    print( '| 03  -----  ACTIVE INDEX                 |' )
    print( '| 04  -----  THUMB PRECISION CLOSED       |' )
    print( '| 05  -----  KEY LATERAL                  |' )
    print( '| 06  -----  FINGER INDEX POINT           |' )
    print( '| 07  -----  MOUSE                        |' )
    print( '| 08  -----  COLUMN                       |' )
    print( '| 09  -----  RELAXED                      |' )
    print( '| 10  -----  NOGRIP                       |' )
    print( '| 11  -----  OPEN                         |' )
    print( '| 12  -----  PRONATE                      |' )
    print( '| 13  -----  SUPINATE                     |' )
    print( '-------------------------------------------' )
    print( '| Press [Q] to quit!                      |' )
    print( '-------------------------------------------' )

    done = False  
    while not done:
        cmd = input( 'Command: ' )
        if cmd.lower() == 'q':
            done = True
        else:
            try:
                idx = int( cmd )
                if idx in range( 0, len( moves ) ):
                    bb3.publish( moves[ idx ] )
            except ValueError:
                pass
    print( 'Bye-bye!' )