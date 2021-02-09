"""
Module for maintaining an Adjusting log across recording sessions.
"""

# System Imports
import json
import logging

# Local Imports
import BrainAtlas
import QtHelperUtils

MODULE_IDENTIFIER = "[AdjustingLogger] "

# Adjusting data input is expected to be in #Turns. Converting that into
# distance metric involves a scaling factor (Screw pitch in our case usually.)
TURN_MULTIPLICATION_FACTOR = 0.25

class TetrodeLog(object):

    """
    Class implementing a JSON based data entry system to track tetrode depths.
    This class contains a dictionary in which, each tetrode is mapped to a set of 3 coordinate values
        - [ML] Medial-Lateral
        - [AP] Anterior-Posterior
        - [DV] Dorsal-Ventral
    """

    def __init__(self, tetrode_list, data_file=None):
        self._tetrode_list = tetrode_list
        self._current_placement = dict()

        data_loaded = self.loadDataFile(data_file)
        if not data_loaded:
            print(MODULE_IDENTIFIER + "Starting entries at default value.")
            for t_num in self._tetrode_list:
                t_str = str(t_num)
                self._current_placement[t_str] = dict()
                self._current_placement[t_str]['coord'] = [BrainAtlas.DEFAULT_ML_COORDINATE, \
                        BrainAtlas.DEFAULT_AP_COORDINATE, BrainAtlas.DEFAULT_DV_COORDINATE]
                # Ideally we would want to keep this as a set but that is not
                # writable to a JSON file directly.
                self._current_placement[t_str]['tags'] = list()
                self._current_placement[t_str]['messages'] = list()

    def getCoordinates(self, tetrode):
        """
        Get the current coordinates for a tetrode.
        """
        if not self.tetrodeExists(tetrode):
            return [0, 0, 0]

        return self._current_placement[tetrode]['coord']

    def tetrodeExists(self, tetrode):
        if tetrode not in self._current_placement:
            logging.warning(MODULE_IDENTIFIER + "Tetrode not found in current placement entry.")
            print(MODULE_IDENTIFIER + "Couldn't find tetrode %s in database"%tetrode)
            # TODO: Maybe add a new entry for this in the future
            return False
        return True

    def getTags(self, tetrode):
        if not self.tetrodeExists(tetrode):
            return []

        return self._current_placement[tetrode]['tags']

    def printMessages(self, tetrode):
        if not self.tetrodeExists(tetrode):
            return []

        print("--------------------------------")
        print("T%s, depth %.2f"%(tetrode, self._current_placement[tetrode]['coord'][2]))
        # Print current depth
        for msg in self._current_placement[tetrode]['messages']:
            print(msg)
        print("--------------------------------")

    def addTags(self, tetrode, new_tags):
        if not self.tetrodeExists(tetrode):
            return []

        for nt in new_tags:
            if nt not in self._current_placement[tetrode]['tags']:
                self._current_placement[tetrode]['tags'].append(nt)

    def addMessage(self, tetrode, new_message):
        if not self.tetrodeExists(tetrode):
            return

        self._current_placement[tetrode]['messages'].append(new_message)

    def updateDepth(self, tetrode, adjustment):
        """
        Update the depth of a given tetrode by the adjustment amount.
        """
        if not self.tetrodeExists(tetrode):
            return

        initial_tetrode_depth = self._current_placement[tetrode]['coord'][2]
        self._current_placement[tetrode]['coord'][2] += adjustment * TURN_MULTIPLICATION_FACTOR
        logging.info(MODULE_IDENTIFIER + "T%s %.2f -> %.2f"%(tetrode, initial_tetrode_depth, \
                self._current_placement[tetrode]['coord'][2]))
        print(MODULE_IDENTIFIER + "T%s %.2f -> %.2f"%(tetrode, initial_tetrode_depth, \
                self._current_placement[tetrode]['coord'][2]))

    def writeDataFile(self, output_filename=None):
        """
        Save the current adjustment coordinates to file.
        """
        if output_filename is None:
            output_filename = QtHelperUtils.get_save_file_name(file_format='Adjusting DB (*.json)', \
                    message='Choose Adjusting Database [Save]')

            if not output_filename:
                return

        try:
            with open(output_filename, 'w') as output_file:
                json.dump(self._current_placement, output_file, indent=4, \
                        sort_keys=True, separators=(',', ': '))
                output_file.close()
            logging.info(MODULE_IDENTIFIER + "Log written to %s"%output_filename)
            print(MODULE_IDENTIFIER + "Log written to %s"%output_filename)
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to write data to file.")
            print(err)

    def loadDataFile(self, data_filename):
        data_loaded = False
        if data_filename is None:
            # Ask the user to select a file from which data should be read.
            # Otherwise create a new file
            data_filename = QtHelperUtils.get_open_file_name(file_format='Adjusting DB (*.json)', \
                    message='Choose Adjusting Database [Load]')

            # User can choose not to select a file here.
            if not data_filename:
                return data_loaded

        try:
            with open(data_filename, 'r') as data_file:
                self._current_placement = json.load(data_file)
                # print(self._current_placement)
            # TODO: Check that all the tetrodes in the current list are there in this database
            data_loaded = True
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to load adjusting data.")
            print(err)

        return data_loaded
