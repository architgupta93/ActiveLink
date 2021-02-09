import os
import time
from datetime import datetime
import logging
from tkinter import filedialog

MODULE_IDENTIFIER = "[DataLogger] "

def getCurrentTime(self):
    return datetime.now().strftime("%H:%M:%S.%f")

class InterruptionLogger(object):
    """
    Logs all messages during ripple/replay disruption into a common log file.
    """

    #TODO: Make a separate thread out of the logger?
    def __init__(self, file_prefix):
        """
        Class constructor. Specify a file prefix to be used for creating logs.

        :file_prefix: Your log file is file_prefix_<DATE>_<TIME>
        """

        time_now = time.gmtime()

    def log(self, message):
        logging.debug(getCurrentTime() + message)
        raise NotImplementedError()

    def exit(self):
        """
        Exit logging and close file
        """
        
        logging.debug(MODULE_IDENTIFIER + "Finished logging at " + getCurrentTime())
        raise NotImplementedError()
