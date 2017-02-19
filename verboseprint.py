from datetime import datetime

PRINT_STATUS='[STATUS]'
PRINT_INFO='[INFO]'
PRINT_WARNING='[!WARNING!]'
PRINT_ERROR='[!ERROR!]'
PRINT_FATAL='[!!FATAL!!]'
PRINT_SUCCESS='[SUCCESS]'
PRINT_FAIL='[FAIL]'
PRINT_OK='[OK]'

class VerbosePrint:
    LOGFILE='data/log'
    QUIET=False

    @classmethod
    def print(cls, string, msg=PRINT_INFO, logfile=None, end='\n', prefix=True):
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


def vprint(string, msg=PRINT_INFO, logfile=None, end='\n', prefix=True):
    VerbosePrint.print(string, msg=msg, logfile=logfile, end=end, prefix=prefix)