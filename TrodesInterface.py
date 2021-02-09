"""
Connection interface to Trodes
"""
import logging
from spikegadgets import trodesnetwork as tn

# Constant declaration
MODULE_IDENTIFIER = "[TrodesInterface] "
LFP_SUBSCRIPTION_ATTRIBUTE = 2014
SPIKE_SUBSCRIPTION_ATTRIBUTE = 1024

class SGClient(tn.AbstractModuleClient):
    """
    Extension of SpikeGadgets client for communicating with Trodes which is
    either recording data in real time or playing back a pre-recorded session.

    William Croughan
    2019/02/11
    """

    timestamps = []

    def __init__(self, name, connection="tcp://127.0.0.1", port=49152):
        """
        Pass on some descriptors to the parent class
        """

        # Call the parent class constructor
        tn.AbstractModuleClient.__init__(self, name, connection, port)
        if (self.initialize() != 0):
            self.recv_quit()
            self.closeConnections()
            error_message = "Could not connect to Trodes!"
            logging.info(MODULE_IDENTIFIER + error_message)
            raise Exception(error_message)
        else:
            logging.info(MODULE_IDENTIFIER + "Initialized connection to Trodes.")
