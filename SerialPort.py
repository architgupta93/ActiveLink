"""
Module for communicating with serial port for communication.
"""

import time
import serial
import logging

BAUDRATE     = 115200
DEFAULT_PORT = '/dev/ttyACM0'
WRITE_DATA   = b'T'
MODULE_IDENTIFIER = "[SerialPort] "

class TriggerPort(serial.Serial):
    """
    Serial port set up to send a single pulse.
    """

    CLASS_IDENTIFIER = "[TriggerPort] "
    def __init__(self, port=DEFAULT_PORT, baud=BAUDRATE):
        self._is_enabled = False
        if port is None:
            port = DEFAULT_PORT
        serial.Serial.__init__(self, port, baud, timeout=0, \
                xonxoff=False, rtscts=True, dsrdtr=True, \
                stopbits=serial.STOPBITS_ONE, \
                bytesize=serial.EIGHTBITS, \
                parity=serial.PARITY_NONE)
        logging.info(MODULE_IDENTIFIER + "Serial port initialized.")

    def sendTriggerPulse(self):
        """
        Send a trigger pulse using the serial channel.
        """
        if self._is_enabled:
            self.write(DEFAULT_MSG)

    def sendTriggerPulseAndTime(self):
        """
        Send a trigger pulse and wait for a response to measure the latency in
        executing a serial pulse.
        """
        pass

    def getStatus(self):
        return self._is_enabled

    def enable(self):
        """
        Enable the serial port (remove pin values from defaults)
        """
        self._is_enabled = True

    def disable(self):
        """
        Disable serial port (allow pin values to be changed by outside input).
        """
        self._is_enabled = False

class BiphasicPort(serial.Serial):
    """
    Serial port set up for biphasic pulse communication
    """

    CLASS_IDENTIFIER = "[BiphasicPort] "
    def __init__(self, port=DEFAULT_PORT, baud=BAUDRATE):
        self._is_enabled = False;
        if port is None:
            port = DEFAULT_PORT
        serial.Serial.__init__(self, port, baud, timeout=0, \
                xonxoff=False, rtscts=False, dsrdtr=False)
        logging.info(self.CLASS_IDENTIFIER + "Serial port initialized.")

    def sendBiphasicPulse(self):
        if self._is_enabled:
            self.write(WRITE_DATA)
            """
            # Worked once with the new box
            self.setDTR(True)
            time.sleep(0.0002)
            # time.sleep(0.0001)
            self.setRTS(True)
            time.sleep(0.0001)
            self.setRTS(False)
            # Some more time is needed here it seems to let the system flush this command
            time.sleep(0.0001)
            self.setDTR(False)
            time.sleep(0.001)

            # The step by step version
            self.setDTR(False)
            self.setRTS(False)
            time.sleep(0.002)
            self.setDTR(False)
            self.setRTS(True)
            time.sleep(0.002)

            self.setRTS(False)
            self.setDTR(True)
            time.sleep(0.002)
            self.setDTR(False)
            time.sleep(0.001)
            self.setRTS(True)

            self.setRTS(True)
            time.sleep(0.0002)
            self.setRTS(False)
            time.sleep(0.0001)
            # self.setRTS(True)
            # time.sleep(0.0002)
            # self.setRTS(False)
            # time.sleep(0.001)
            # Never worked  with the new box
            self.setDTR(True)
            time.sleep(0.0002)
            self.setDTR(False)
            time.sleep(0.0001)
            self.setRTS(True)
            time.sleep(0.0002)
            self.setRTS(False)
            time.sleep(0.001)
            """
            logging.info(MODULE_IDENTIFIER + "Biphasic pulse delivered.")
        else:
            logging.info(MODULE_IDENTIFIER + "WARNING! Attempted Biphasic pulse without enabling device! Ignoring!")

    def getStatus(self):
        return self._is_enabled

    def enable(self):
        """
        Enable the serial port (remove pin values from defaults)
        """
        self._is_enabled = True
        if not self.is_open:
            self.open()

    def disable(self):
        """
        Disable serial port (allow pin values to be changed by outside input).
        """
        self._is_enabled = False
        if self.is_open:
            self.close()
