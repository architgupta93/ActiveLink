"""
Extending Thread class to have a stopping event which we can set to stop the
thread and safely get all the data that the thread has at the moment.
"""

import threading
import multiprocessing

class StoppableThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def req_stop(self):
        return self._stop_event.is_set()

    def reset(self):
        self._stop_event.clear()

    def join(self, timeout=None):
        # For all implementations of this class, need to extend the join method
        # to log the time at which thread ended.
        self.stop()
        threading.Thread.join(self, timeout)

class StoppableProcess(multiprocessing.Process):
    """
    A process which can be stopped by calling STOP function on it
    """

    def __init__(self):
        """TODO: to be defined1. """
        multiprocessing.Process.__init__(self, daemon=True)
        self._stop_event = multiprocessing.Event()

    def stop(self):
        """
        Allows setting the _stop_event flag that can be checked to stop thread
        execution
        """
        self._stop_event.set()

    def req_stop(self):
        return self._stop_event.is_set()

    def reset(self):
        self._stop_event.clear()

    def join(self, timeout=None):
        # For all implementations of this class, need to extend the join method
        # to log the time at which thread ended.
        self.stop()
        multiprocessing.Process.join(self, timeout)
