import sys

# PLOT FUNCTIONS
import matplotlib
matplotlib.use('QT5Agg')

# TIME FUNCTIONS
import ctypes

platform = sys.platform
hifi_timing = True
if platform == 'linux':
    LIBC = ctypes.CDLL('libc.so.6') # for .......just linux?
elif platform == 'darwin':
    LIBC = ctypes.CDLL('libc.dylib') # mac version of c lib for clocking
elif platform == 'win32':
    LIBC = ctypes.CDLL('windll.kernel32') # for windows dweebs
else:
    hifi_timing = False
    raise RuntimeWarning( 'Unsupported OS for precision timing!' )

if hifi_timing:
    class Timespec( ctypes.Structure ):
        _fields_ = [ ('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long) ]
    LIBC.nanosleep.argtypes = [ctypes.POINTER(Timespec), ctypes.POINTER(Timespec)]
    nanosleep_req = Timespec()
    nanosleep_rem = Timespec()

    def ns_sleep( ns ):
        nanosleep_req.tv_sec = int( ns / 1e9 )
        nanosleep_req.tv_nsec = int( ns % 1e9 )
        LIBC.nanosleep( nanosleep_req, nanosleep_rem )

    def us_sleep( us ):
        LIBC.usleep( int( us ) )
else:
    import time
    def ns_sleep( ns ): time.sleep( ns * 1e-9 )
    def us_sleep( us ): time.sleep( us * 1e-6 )
