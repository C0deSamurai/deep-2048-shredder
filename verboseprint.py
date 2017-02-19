from datetime import datetime

PRINT_STATUS='[STATUS]'
PRINT_INFO='[INFO]'
PRINT_WARNING='[!WARNING!]'
PRINT_ERROR='[!ERROR!]'
PRINT_FATAL='[!!FATAL!!]'
PRINT_SUCCESS='[SUCCESS]'
PRINT_FAIL='[FAIL]'
PRINT_OK='[OK]'

LOGFILE='data/log'

def vprint(string, msg=PRINT_INFO, logfile=None, end='\n', prefix=True):
    now = datetime.now()
    datestr = now.replace(microsecond=0)
    if prefix:
        outstr = "{} {}: {}".format(msg, datestr, string)
    else:
        outstr = string
    print(outstr, end=end)
    with open(LOGFILE if logfile is None else logfile, 'a+') as outfile:
        outfile.write(outstr + end)