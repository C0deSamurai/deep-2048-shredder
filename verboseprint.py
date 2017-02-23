from datetime import datetime

PRINT_STATUS='[STATUS]'
PRINT_INFO='[INFO]'
PRINT_WARNING='[!WARNING!]'
PRINT_ERROR='[!ERROR!]'
PRINT_FATAL='[!!FATAL!!]'
PRINT_SUCCESS='[SUCCESS]'
PRINT_FAIL='[FAIL]'
PRINT_OK='[OK]'
PRINT_DEBUG='[DEBUG]'

class VerbosePrint:
    LOGFILE='data/log'
    QUIET=False
    DEBUG=False

    @classmethod
    def print(cls, string, msg=PRINT_INFO, logfile=None, end='\n', prefix=True, debug=False):
        if cls.DEBUG == True:
            if debug == True:
                msg = PRINT_DEBUG
        if cls.QUIET == True:
            return
        
        now = datetime.now()
        datestr = now.replace(microsecond=0)
        if prefix:
            outstr = "{} {}: {}".format(msg, datestr, string)
        else:
            outstr = string
        print(outstr, end=end)
        with open(cls.LOGFILE if logfile is None else logfile, 'a+') as outfile:
            outfile.write(outstr + end)


def vprint(string, msg=PRINT_INFO, logfile=None, end='\n', prefix=True, debug=False):
    VerbosePrint.print(string, msg=msg, logfile=logfile, end=end, prefix=prefix, debug=debug)

def vprint_np(string, logfile=None, end='\n', debug=False):
    """
    Same as vprint, but never adds the prefix (_np for 'no prefix')
    """
    VerbosePrint.print(string, logfile=logfile, end=end, prefix=False, debug=debug)