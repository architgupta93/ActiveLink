# System imports
import sys
import threading
import time
import ctypes
import numpy as np
import logging
import cProfile
import collections
from multiprocessing import Queue, RawArray, Condition
from multiprocessing import Pipe, Lock, Event, Value

# PyQt imports
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QMessageBox, QDialog
from PyQt5 import QtCore

# Local imports
import Logger
import BrainAtlas
import SerialPort
import QtHelperUtils
import Configuration
import Visualization
import ThreadExtension
import RippleAnalysis
import PositionAnalysis
import TrodesInterface
import RippleDefinitions as RiD

MODULE_IDENTIFIER = "[OnlineInterruption] "
# User selection choices for what they want to see on the screen

# Configuration for looking at spikes and fields
"""
DEFAULT_LFP_CHOICE      = False
DEFAULT_SPIKES_CHOICE   = True
DEFAULT_POSITION_CHOICE = True
DEFAULT_FIELD_CHOICE    = True
DEFAULT_STIMULATION_CHOICE = False
DEFAULT_CALIBRATION_CHOICE = False
DEFAULT_BAYESIAN_CHOICE = True
"""

# Configuration for LFP and adjusting
DEFAULT_CLUSTER_SELECTION_CHOICE = True     # Decides whether user will get a pop-up to choose hand-clustering
DEFAULT_CLUSTER_MODEL_CHOICE = True         # Decides whether user will get a pop-up to choose an existing clustering model.

DEFAULT_LFP_CHOICE      = False
DEFAULT_POSITION_CHOICE = True
DEFAULT_STIMULATION_CHOICE = False
DEFAULT_ADJUSTING_CHOICE = False
DEFAULT_LINEAR_TRACK_CHOICE = True

# Create a dictionary for the default choices
D_USER_PROCESSING_CHOICES = dict()
D_USER_PROCESSING_CHOICES['lfp']      = DEFAULT_LFP_CHOICE
D_USER_PROCESSING_CHOICES['position'] = DEFAULT_POSITION_CHOICE
D_USER_PROCESSING_CHOICES['stim']     = DEFAULT_STIMULATION_CHOICE
D_USER_PROCESSING_CHOICES['adjusting']= DEFAULT_ADJUSTING_CHOICE
D_USER_PROCESSING_CHOICES['linear']   = DEFAULT_LINEAR_TRACK_CHOICE
PROCESSING_ARGS = [ "Local Field Potential (LFP)", \
        "Position Data", \
        "Stimulation", \
        "Adjusting", \
        "Linear Track"]

# Choices in functionality
DEFAULT_SERIAL_ENABLED = False
DEFAULT_STIM_MODE_MANUAL_ENABLED = False
DEFAULT_STIM_MODE_POSITION_ENABLED = False
DEFAULT_STIM_MODE_VELOCITY_ENABLED = False
DEFAULT_STIM_MODE_RIPPLE_ENABLED = False
DEFAULT_STIM_MODE_SPIKE_DENSITY_ENABLED = False
DEFAULT_UPDATE_RIPPLE_STATS = True

# Define trigger zones as a list of centers and radii. 
# N_TRIGGER_ZONES = 2
# DEFAULT_TRIGGER_ZONES = [(1, 12, 1.5), (11, 7.5, 1.5)]
N_TRIGGER_ZONES = 2
DEFAULT_TRIGGER_ZONES = [(2.5, 12, 1.5), (11, 7.5, 1.5)]

# Periodically check if clustering is done in order to start processing the
# clustering data.
CLUSTERING_WAIT_TIME = 1

class StimulationSynchronizer(ThreadExtension.StoppableProcess):
    """
    Waits for a stimulation events to be detected and processes downstream changes for
    analyzing spike contents.
    """

    # Wait for 10ms while checking if the event flag is set.
    _EVENT_TIMEOUT = 1.0
    _SPIKE_BUFFER_SIZE = 200
    CLASS_IDENTIFIER = "[StimulationSynchronizer] "

    def __init__(self, spike_listener, position_estimator, place_field_handler, swr_sync_event, \
            sde_sync_event, sg_client=None, serial_port=None):
        """
        Processd for synchronizing all measurement modalities and generate
        stimulation signal based on them.
        """
        ThreadExtension.StoppableProcess.__init__(self)
        self._swr_sync_event = swr_sync_event
        self._sde_sync_event = sde_sync_event
        self._spike_buffer = collections.deque(maxlen=self._SPIKE_BUFFER_SIZE)
        self._spike_histogram = collections.Counter()

        # Look at the spike information if it is available
        if spike_listener is not None:
            self._spike_buffer_connection = spike_listener.get_spike_buffer_connection()
        else:
            self._spike_buffer_connection = None

        # Look at position information if it is available
        if position_estimator is not None:
            self._position_buffer_connection = position_estimator.get_position_buffer_connection()
        else:
            self._position_buffer_connection = None

        self._place_field_handler = place_field_handler
        self._sg_client = sg_client
        self._position_access = Lock()
        self._spike_access = Lock()
        # TODO: This functionality should be moved to the parent class
        self._enable_synchrnoizer = Lock()
        self._clusters_of_interest = [Configuration.EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_A[:], \
                Configuration.EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_B[:]]
        print(self._clusters_of_interest)
        self._is_disabled = Value("b", True)
        self._position_trig_is_disabled = Value("b", True)
        self._velocity_trig_is_disabled = Value("b", True)
        self._swr_trig_is_disabled = Value("b", True)
        self._sde_trig_is_disabled = Value("b", True)

        # Allocate an array to store the current stimulation zones. Each
        # stimulation zone is characterized by a center coordinate and the size
        # of the stimulation zone.
        self._trigger_lock = Lock()
        self._n_trigger_zones = N_TRIGGER_ZONES
        self._shared_trigger_zones = RawArray(ctypes.c_double, self._n_trigger_zones * 3)
        self._trigger_zones = np.reshape(np.frombuffer(self._shared_trigger_zones, dtype=ctypes.c_double), \
                (self._n_trigger_zones, 3))

        # Position data at the time ripple is triggered
        self._pos_x = -1
        self._pos_y = -1
        self._most_recent_speed = 0
        self._inside_stim_zone = False
        self._most_recent_pos_timestamp = 0
        self._serial_port = None
        try:
            self._serial_port = SerialPort.BiphasicPort(serial_port)
        except Exception as err:
            logging.warning(self.CLASS_IDENTIFIER + "Unable to open Serial port.")
            print(err)

        # Calling set up functions
        self.reset_trigger_zones()
        logging.info(self.CLASS_IDENTIFIER + "Started Stimulation Synchronization thread.")

    def reset_trigger_zones(self):
        """
        Reset the trigger zone to its default value.
        """
        with self._trigger_lock:
            self._trigger_zones[0,:] = DEFAULT_TRIGGER_ZONES[0][:]
            self._trigger_zones[1,:] = DEFAULT_TRIGGER_ZONES[1][:]

    def get_trigger_zones(self):
        # TODO: Check to make sure that this works and does not return empty arrays.
        return np.copy(self._trigger_zones)

    def enableSerial(self):
        if self._serial_port is not None:
            self._serial_port.enable()
            print(MODULE_IDENTIFIER + "Serial port enabled.")
        else:
            print(MODULE_IDENTIFIER + "Serial port not found.")

    def disableSerial(self):
        if self._serial_port is not None:
            self._serial_port.disable()

    def testContinuousStim(self):
        """
        Stimulate irrespective of the recording events and conditions for
        duration and period specified by the configuration.
        """
        self.startManualStimulation()

    def testSingleStim(self):
        """
        Stimulate irrespective of the recording events and conditions ONCE
        """
        # First make sure that the serial port is well defined and enabled
        if self._serial_port is not None:
            if self._serial_port.getStatus():
                self._serial_port.sendBiphasicPulse()
                logging.info(self.CLASS_IDENTIFIER + time.strftime("Delivered STIM at %Y.%m%.d %H:%M:%S"))
            else:
                QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Port disabled!')
        else:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Port undefined!')

    def startManualStimulation(self):
        """
        Stimulate irrespective of the recording events and conditions for
        duration and period specified by the configuration.
        """
        # First make sure that the serial port is well defined and enabled
        if self._serial_port is not None:
            if self._serial_port.getStatus():
                stim_start_time = time.time()
                current_time = time.time()
                while (current_time - stim_start_time < Configuration.MANUAL_STIM_DURATION):
                    self._serial_port.sendBiphasicPulse()
                    time.sleep(Configuration.MANUAL_STIM_INTER_PULSE_INTERVAL)
                    logging.info(self.CLASS_IDENTIFIER + time.strftime("Delivered STIM at %Y.%m%.d %H:%M:%S"))
                    current_time = time.time()
            else:
                QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Port disabled!')
        else:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Port undefined!')

    def enable(self):
        self._enable_synchrnoizer.acquire()
        self._is_disabled.value = False
        logging.info(self.CLASS_IDENTIFIER + "Stimulation ENABLED.")
        self._enable_synchrnoizer.release()

    def enable_stimulation_trigger(self):
        """
        Enable the process to look at sharp-wave ripple trigger and stimulate based on it.
        """
        with self._enable_synchrnoizer:
            self._is_disabled.value = False
            self._swr_trig_is_disabled.value = False
            logging.info(self.CLASS_IDENTIFIER + "Sharp-Wave Ripple trigger ENABLED.")

    def disable_stimulation_trigger(self):
        """
        Disable the process from looking at sharp-wave ripple trigger and
        stimulating based on it.
        """
        with self._enable_synchrnoizer:
            self._swr_trig_is_disabled.value = True
            logging.info(self.CLASS_IDENTIFIER + "Sharp-Wave Ripple trigger DISABLED.")

    def enable_position_trigger(self):
        """
        Check to see that the current location is in the trigger zone before stimulating.
        """
        with self._enable_synchrnoizer:
            self._is_disabled.value = False
            self._position_trig_is_disabled.value = False
            logging.info(self.CLASS_IDENTIFIER + "Position trigger ENABLED.")

    def disable_position_trigger(self):
        """
        Disable all position (trigger zone) checks for stimulation.
        """
        with self._enable_synchrnoizer:
            self._position_trig_is_disabled.value = True
            logging.info(self.CLASS_IDENTIFIER + "Position trigger DISABLED.")

    def enable_velocity_trigger(self):
        """
        Enable trigger based on velocity (below cutoff) condition.
        """
        with self._enable_synchrnoizer:
            self._is_disabled.value = False
            self._velocity_trig_is_disabled.value = False
            logging.info(self.CLASS_IDENTIFIER + "Velocity trigger ENABLED.")

    def enable_spike_density_trigger(self):
        """
        Enable trigger based on spike density events.
        TODO: The event has to be supplied externally, no calculations will be done for it here.
        """
        with self._enable_synchrnoizer:
            self._is_disabled.value = False
            self._sde_trig_is_disabled.value = False
            logging.info(self.CLASS_IDENTIFIER + "Spike Density trigger ENABLED.")

    def disable_spike_density_trigger(self):
        """
        Disable spike density checks for stimulation.
        """
        with self._enable_synchrnoizer:
            self._sde_trig_is_disabled.value = True
            logging.info(self.CLASS_IDENTIFIER + "Spike Density trigger DISABLED.")

    def disable(self):
        self._enable_synchrnoizer.acquire()
        self._is_disabled.value = True
        logging.info(self.CLASS_IDENTIFIER + "Stimulation DISABLED.")
        self._enable_synchrnoizer.release()

    def fetch_current_velocity(self):
        """
        Get recent velocity (and position) and use that to determine if the
        animal was running when the current ripple was detected.
        """
        while not self.req_stop():
            if self._position_buffer_connection.poll():
                position_data = self._position_buffer_connection.recv()
                with self._position_access:
                    current_zone_check = self._inside_stim_zone
                    self._most_recent_pos_timestamp = position_data[0]
                    self._most_recent_speed = position_data[3]
                    self._pos_x = position_data[1]
                    self._pos_y = position_data[2]

                    # For each of the trigger zones, check if the animal is inside one of them
                    self._inside_stim_zone = False
                    for t_zone in range(self._n_trigger_zones):
                        tz_dx = self._trigger_zones[t_zone,0] - self._pos_x
                        tz_dy = self._trigger_zones[t_zone,1] - self._pos_y
                        tz_r = self._trigger_zones[t_zone,2]

                        in_tzone = (tz_dx * tz_dx) + (tz_dy * tz_dy) < tz_r * tz_r
                        if in_tzone:
                            self._inside_stim_zone = True
                            if not current_zone_check:
                                print(MODULE_IDENTIFIER + "Animal entered trigger zone.")
                            break

                    if current_zone_check and (not self._inside_stim_zone):
                        print(MODULE_IDENTIFIER + "Animal exitted trigger zone.")
            else:
                time.sleep(0.005)

    def fetch_most_recent_spike(self):
        """
        Get the most recent spike and put it in the rotating spike buffer
        (keeps track of the last self._SPIKE_BUFFER_SIZE spikes.)
        """
        while not self.req_stop():
            if self._spike_buffer_connection.poll():
                # NOTE: spike_data received here is a tuple (cluster identity, trodes timestamp)
                spike_data = self._spike_buffer_connection.recv()
                with self._spike_access:
                    if len(self._spike_buffer) == self._SPIKE_BUFFER_SIZE:
                        removed_spike = self._spike_buffer.popleft()
                        self._spike_histogram[removed_spike[0]] -= 1
                    spike_cluster = spike_data[0]
                    if (spike_cluster in self._clusters_of_interest[0]) or \
                            (spike_cluster in self._clusters_of_interest[1]):
                        # NOTE: If this starts taking too long, can switch to default dictionary
                        self._spike_buffer.append(spike_data)
                        self._spike_histogram[spike_cluster] += 1
            else:
                # NOTE: Making the thread sleep for 5ms might not hurt but we
                # will have to find out.
                time.sleep(0.005)

    def run(self):
        # Create a thread that fetches and keeps track of the last few spikes.
        if self._spike_buffer_connection is not None:
            spike_fetcher = threading.Thread(name="SpikeFetcher", daemon=True, \
                    target=self.fetch_most_recent_spike)
            spike_fetcher.start()
        else:
            spike_fetcher = None

        if self._position_buffer_connection is not None:
            velocity_fetcher = threading.Thread(name="VelocityFetcher", daemon=True, \
                    target=self.fetch_current_velocity)
            velocity_fetcher.start()
        else:
            velocity_fetcher = None

        while not self.req_stop():
            if self._serial_port is None:
                QtHelperUtils.display_warning("Unable to access serial port. Stimulation thread exitting!")
                break

            # Check if the process has been enabled
            with self._enable_synchrnoizer:
                current_system_state = self._is_disabled.value
                current_position_trig_state = self._position_trig_is_disabled.value
                current_velocity_trig_state = self._velocity_trig_is_disabled.value
                current_swr_trig_state = self._swr_trig_is_disabled.value
                current_sde_trig_state = self._sde_trig_is_disabled.value

            if current_system_state:
                logging.debug(self.CLASS_IDENTIFIER + "Process sleeping")
                time.sleep(0.002)
                continue

            # Check if position trigger is set. If it is, then check if the
            # current position is in the trigger zone.

            if not self._serial_port.getStatus():
                print(self.CLASS_IDENTIFIER + "Enabling Serial port for ripple disruption")
                self._serial_port.enable()

            with self._position_access:
                if current_position_trig_state:
                    is_within_stim_zone = True
                else:
                    is_within_stim_zone = self._inside_stim_zone 

                if current_velocity_trig_state:
                    is_within_velocity_th = True
                else:
                    is_within_velocity_th = self._most_recent_speed < RiD.MOVE_VELOCITY_THRESOLD

                position_constraint_matched = is_within_velocity_th and is_within_stim_zone

            if not current_swr_trig_state:
                logging.debug(self.CLASS_IDENTIFIER + "Waiting for ripple trigger...")
                with self._swr_sync_event:
                    thread_notified = self._swr_sync_event.wait(self._EVENT_TIMEOUT)
            elif not current_sde_trig_state:
                # TODO: Currently, nothing is supplied in for sde_sync_event so
                # this will lead to an error. FIXME
                logging.debug(self.CLASS_IDENTIFIER + "Waiting for spike-density trigger...")
                with self._swr_sync_event:
                    thread_notified = self._sde_sync_event.wait(self._EVENT_TIMEOUT)

            if thread_notified and position_constraint_matched:
                self._serial_port.sendBiphasicPulse()

                print(self.CLASS_IDENTIFIER + time.strftime("Ripple triggered at... %Y%m%d_%H%M%S"))
                logging.info(self.CLASS_IDENTIFIER + time.strftime("Ripple triggered at ... %Y%m%d_%H%M%S"))

        logging.info(self.CLASS_IDENTIFIER + "Main process exited.")
        if spike_fetcher is not None:
            spike_fetcher.join()
        if velocity_fetcher is not None:
            velocity_fetcher.join()

        logging.info(self.CLASS_IDENTIFIER + "Helper threads exited.")
        logging.info(MODULE_IDENTIFIER + "Stimulation event synchronizer Stopped")

class CommandWindow(QMainWindow):
    """
    Parent window for running all the programs
    """

    def __init__(self, arguments):
        """
        Class constructor for the main application window
        """
        QMainWindow.__init__(self)
        self.setWindowTitle('ActiveLink')
        self.statusBar().showMessage('Connect to SpikeGadgets.')
        self.stim_mode_position = None
        self.stim_mode_ripple = None
        self.update_ripple_stats_menu = None
        self.setupMenus()

        # TODO: None of the thread classes have any clean up at the end... TBD
        if __debug__:
            # Create code profiler
            self.code_profiler = cProfile.Profile()
            profile_prefix = "stimulation_profile"
            self.profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")

        # Tetrode info fields
        self.n_units = 0
        self.n_tetrodes = 0
        self.cluster_identity_map = dict()
        self.tetrodes_of_interest = None

        # Shared memory buffers for passing information across threads
        self.shared_posterior_buffer = None
        self.shared_posterior_bin_times = None
        self.shared_raw_lfp_buffer = None
        self.shared_ripple_buffer = None
        self.shared_place_fields = None
        self.shared_spk_times = None
        self.shared_spk_rates = None

        # Shared arrays for stimulus calibration
        self.shared_calib_plot_times = None
        self.shared_calib_plot_counts = None

        # Trodes connection
        self.sg_client = None
        self.data_streaming = False

        # Serial connection
        self.serial_port = None

        # Synchronization conditions across threads
        self.decoding_done      = Condition()
        self.swr_trig_condition = Condition()
        self.sde_trig_condition = Condition()
        self.show_swr_trigger   = [Event(), Lock()]
        self.show_sde_trigger   = [Event(), Lock()]
        self.calib_trigger      = [Event(), Lock()]

        # Initialize containers for all the thread processors
        # NOTE: spike_sorter is a special field. It doesn't spawn a new thread.
        # Depending on user choices, it will either be a pointer to
        # spike_listener (which is getting cluster identities from Trodes), or
        # it will be the online clustering mechanism which will take spikes,
        # cluster them and then supply cluster identities.

        self.calib_plot          = None
        self.lfp_listener        = None
        self.stimulation_trigger = None
        self.spike_listener      = None
        self.online_clustering   = None
        self.spike_sorter        = None
        self.ripple_detector     = None
        self.position_estimator  = None
        self.sde_estimator       = None
        self.place_field_handler = None
        self.bayesian_estimator  = None
        self.graphical_interface = None

        # Shared condition which will be set when clustering is complete. At
        # this point, we can start building place fields and decoding.
        self.check_for_clustering = False
        self._clustering_finished  = [Event(), Lock()]
        self.session_unit_list =  None

        # Analysis mode set for linear track
        self._is_linear_track    = False

        # Brain Atlas interface
        self.brain_atlas = BrainAtlas.WebAtlas()

        # Put all the processes in a list so that we don't have to deal with
        # each of them by name when starting/stopping streaming.

        # UPDATE 2021-01-19: Added another set of processes that are delayed.
        # They do not start at startup but intead, are started after receiving
        # a trigger condition from the active processes.
        self.active_processes    = list()
        self.delayed_processes   = list()
        self.user_processing_choices = dict()

        # Setup child Qt objects to be called later using menu functions
        self._ripple_preference_menu = None
        self._ripple_reference_tet = None
        self._ripple_baseline_tet = None
        if arguments.reference_tetrode:
            self._ripple_reference_tet = arguments.reference_tetrode-1

        if arguments.baseline_tetrode:
            self._ripple_baseline_tet = arguments.baseline_tetrode-1

        # Launch the main graphical interface as a widget
        self.setGeometry(100, 100, 480, 50)

        """
        # The 2 lines below remove the CLOSE button on the window.
        # enable custom window hint
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)

        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        """

    def closeEvent(self, event):
        self.disconnectAndQuit()

    # Functions for saving data
    def saveDisplaySnapshot(self):
        if self.graphical_interface is not None:
            # Save the current display frame
            saved_file = self.graphical_interface.saveDisplay()
            if saved_file:
                self.statusBar().showMessage("Display snapshot saved to disk")

    def saveAdjustingLog(self):
        if self.graphical_interface is not None:
            self.graphical_interface.autosaveAdjustingLog()

    def saveRippleStats(self):
        if self.ripple_detector is not None:
            # Save the current ripple stats to file
            ripple_stats_saved = self.ripple_detector.save_ripple_stats()
        else:
            ripple_stats_saved = False 

        if ripple_stats_saved:
            self.statusBar().showMessage("Ripple Stats saved.")

    # Functions for loading data
    def loadRippleStats(self):
        if self.ripple_detector is None:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Ripple detector not setup to load stats.')
            return

        # Get the name of the place-field file
        ripple_stats_filename = QtHelperUtils.get_open_file_name(file_format="Ripple Stats (*.npz)", message="Select ripple stats file")
        if ripple_stats_filename:
            ripple_stats_loaded = self.ripple_detector.load_ripple_stats(ripple_stats_filename)
        else:
            ripple_stats_loaded = False

        if ripple_stats_loaded:
            self.statusBar().showMessage("Ripple Stats loaded.")

    ############################# STIMULATION TRIGGERS #############################
    # Set up the different stimulation methods here. The three different
    # methods that we plan to use right now are:
    #   1. Manual Trigger - Trigger for a fixed duration immediately after the
    #       menu is selected.
    #   2. Position Trigger - Allow the user to select a position (trigger zone)
    #       within which stimulation will be activated.
    #   3. Velocity Trigger - Allow the user to apply a velocity thresold to
    #       the stimulation trigger.
    #   4. Ripple Trigger - The good old, trigger on Sharp-Wave ripples in the
    #       Hippocampus. It can be tricky to eliminate noise, whose broadband
    #       power can also be seen in the ripple band.
    #   5. Spike-Density Trigger: Look at spike density events to trigger.
    ################################################################################

    def manualStimTrigger(self):
        if self.stimulation_trigger is not None:
            self.statusBar().showMessage("Starting test stimulation...")
            self.stimulation_trigger.startManualStimulation()
            self.statusBar().showMessage("Test stimulation Finished.")
        else:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + "Stimulation thread not initialized.")

    def positionStimTrigger(self, state):
        if self.stimulation_trigger is None:
            return

        if state:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enable trigger zones?')
            if user_response == QMessageBox.Ok:
                self.stimulation_trigger.enable_position_trigger()
        else:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Disable trigger zones?')
            if user_response == QMessageBox.Ok:
                self.stimulation_trigger.disable_position_trigger()

    def velocityStimTrigger(self, state):
        if state:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enable velocity thresholded trigger?')
            if user_response == QMessageBox.Ok:
                self.stimulation_trigger.enable_velocity_trigger()
        else:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Disable velocity thresholded trigger?')
            if user_response == QMessageBox.Ok:
                self.stimulation_trigger.disable_velocity_trigger()

    def rippleStimTrigger(self, state):
        if state:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enable ripple trigger?')
            if user_response == QMessageBox.Ok: 
                self.stimulation_trigger.enable_stimulation_trigger()
        else:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Disable ripple trigger?')
            if user_response == QMessageBox.Ok: 
                self.stimulation_trigger.disable_stimulation_trigger()

    def spikeStimTrigger(self, state):
        if state:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enable Spike-Density trigger?')
            if user_response == QMessageBox.Ok:
                self.stimulation_trigger.enable_spike_density_trigger()
        else:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Disable Spike-Density trigger?')
            if user_response == QMessageBox.Ok:
                self.stimulation_trigger.disable_spike_density_trigger()

    # Preference selection
    def collectSWRPreferences(self):
        if self._ripple_preference_menu is None:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + "LFP Stream not setup.")
            return
        user_ok = self._ripple_preference_menu.exec_()
        if (user_ok == QDialog.Accepted):
            ripple_reference, ripple_baseline = self._ripple_preference_menu.getIdxs()
            self.ripple_detector.set_ripple_reference(ripple_reference)
            self.ripple_detector.set_ripple_baseline(ripple_baseline)

    def setUpdateRippleStats(self, state):
        if self.ripple_detector is None:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + "Ripple detection not setup.")
            return
        self.ripple_detector.set_ripple_stat_update(state)
        if state:
            self.statusBar().showMessage("Started updating SWR stats.")
        else:
            self.statusBar().showMessage("Stopped updating SWR stats.")

    def zoomInClustering(self):
        if self.graphical_interface is not None:
            self.graphical_interface.scale_cluster_zoom(0.75)

    def toggleClusterColors(self, state):
        """
        Toggle cluster coloring in spike display.
        """

        if self.graphical_interface is not None:
            self.graphical_interface.set_cluster_colors(state)

    def clearSpikeDisplay(self):
        """
        Clear spikes currently in display
        """

        if self.graphical_interface is not None:
            self.graphical_interface.clear_clusters()

    def zoomOutClustering(self):
        if self.graphical_interface is not None:
            self.graphical_interface.scale_cluster_zoom(1.5)

    def resetSWRStats(self):
        if self.ripple_detector is None:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + "Ripple detection not setup.")
            return
        user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Reset ripple stats?')
        if user_response == QMessageBox.Ok: 
            self.ripple_detector.reset_ripple_stats()

    ############################# SERIAL FUNCTIONALITY #############################
    # Add functions that let you access and test the serial port in a
    # convenient way. This allows you to safely enable/disable the serial port
    # and test the stimulating electrode's status by sending a single pulse OR
    # a series of pulses.
    ################################################################################
    # Select serial port to trigger stimulation on
    def selectSerialPort(self):
        """
        Select a port to enable serial communication on.
        TODO: Add the ability to renew the serial connection with the new port
        if a selection is made at runtime.
        """
        selection_widget = QtHelperUtils.PortSelectionDialog()
        user_ok, selected_port = selection_widget.exec_()
        if user_ok == QDialog.Accepted:
            print(MODULE_IDENTIFIER + "Selected port %s for stimulation."% selected_port)
            self.serial_port = selected_port

    # Set up the serial port
    def enableSerialPort(self, state):
        if self.stimulation_trigger is None:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Stimulation thread disabled. Enable for serial COM.')
            self.enable_serial_port.setChecked(False)
            return;

        # TODO: To the information statement above, add a line telling which
        # port is currently being used.
        if state:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enable serial port?')
            if user_response == QMessageBox.Ok: 
                self.stimulation_trigger.enableSerial()
        else:
            user_response = QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Disable serial port?')
            if user_response == QMessageBox.Ok: 
                self.stimulation_trigger.disableSerial()

    def testSingleSerialPulse(self):
        if self.stimulation_trigger is not None:
            self.stimulation_trigger.testSingleStim()
            self.statusBar().showMessage("Sent single Biphasic pulse.")

    def testContinuousSerialPulse(self):
        if self.stimulation_trigger is not None:
            self.stimulation_trigger.testContinuousStim()
            self.statusBar().showMessage("Finished Biphasic pulse stream.")

    # Setting plot areas
    def plotRippleHistory(self):
        """
        Plot the history of sharp wave ripples for the current session. This
        also allows you to compare sharp-wave ripples at different electrodes
        at the present time.
        """
        if self.graphical_interface is not None:
            try:
                self.graphical_interface.display_past_ripples(self._ripple_reference_tet)
            except Exception as err:
                QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Failed to plot SWR history.')
                print(err)
        else:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Initialize processing threads to see SWR history.')

    def showBrainAtlas(self):
        if self.graphical_interface is not None:
            self.graphical_interface.showTetrodeInBrain()
        else:
            user_response, coordinates, view_selection = QtHelperUtils.BrainCoordinateWidget().exec_()
            if user_response == QDialog.Accepted:
                if 0 in view_selection:
                    self.brain_atlas.getCoronalImage(*coordinates)
                if 1 in view_selection:
                    self.brain_atlas.getSagittalImage(*coordinates)
                if 2 in view_selection:
                    self.brain_atlas.getHorizontalImage(*coordinates)

    def plotBayesianEstimate(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Show ripple stats (current as well as history) for tetrodes
    def showRippleStats(self):
        if self.ripple_detector is not None:
            self.ripple_detector.show_ripple_stats()

    def disconnectAndQuit(self):
        if QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Quit Program?') == QMessageBox.Cancel:
            return

        if self.graphical_interface is not None:
            self.graphical_interface.kill_gui()
            if self.data_streaming:
                self.stopThreads()

            try:
                if self.sg_client is not None:
                    self.sg_client.closeConnections()
            except Exception as err:
                print(MODULE_IDENTIFIER + "Unable to close connection to Trodes. Not that it won't throw seg fault in your face anyways!")
                print(err)

        print(MODULE_IDENTIFIER + "Program finished. Exiting.")
        qApp.quit()

    def setupMenus(self):
        # Set up the menu bar
        menu_bar = self.menuBar()

        # File menu - Save, Load (Processed Data), Quit
        file_menu = menu_bar.addMenu('&File')

        connect_action = file_menu.addAction('&Connect')
        connect_action.setShortcut('Ctrl+N')
        connect_action.setStatusTip('Connect SpikeGadgets')
        connect_action.triggered.connect(self.connectSpikeGadgets)

        stream_action = file_menu.addAction('S&tream')
        stream_action.setShortcut('Ctrl+T')
        stream_action.setStatusTip('Stream data')
        stream_action.triggered.connect(self.streamData)

        self.freeze_action = QAction('Free&ze', self, checkable=True)
        self.freeze_action.setShortcut('Ctrl+Z')
        self.freeze_action.setStatusTip('Freeze display')
        self.freeze_action.setChecked(False)
        self.freeze_action.triggered.connect(self.freezeStream)
        file_menu.addAction(self.freeze_action)

        # =============== SAVE MENU =============== 
        save_menu = file_menu.addMenu('&Save')
        save_display_snapshot_action = QAction('&Display', self)
        save_display_snapshot_action.setStatusTip('Save display snapshot')
        save_display_snapshot_action.triggered.connect(self.saveDisplaySnapshot)
        save_display_snapshot_action.setShortcut('Ctrl+S')

        save_adjusting_log_action = QAction('&Adjusting', self)
        save_adjusting_log_action.setStatusTip('Save current adjusting coordinates')
        save_adjusting_log_action.triggered.connect(self.saveAdjustingLog)

        save_ripple_stats_action = QAction('&Ripple Stats', self)
        save_ripple_stats_action.setStatusTip('Save current SWR stats to file')
        save_ripple_stats_action.triggered.connect(self.saveRippleStats)

        save_menu.addAction(save_display_snapshot_action)
        save_menu.addAction(save_adjusting_log_action)
        save_menu.addAction(save_ripple_stats_action)

        # =============== LOAD MENU =============== 
        open_menu = file_menu.addMenu('&Load')
        load_ripple_stats_action = QAction('&Ripple Stats', self)
        load_ripple_stats_action.setStatusTip('Load Ripple Stats from a file')
        load_ripple_stats_action.triggered.connect(self.loadRippleStats)
        open_menu.addAction(load_ripple_stats_action)

        quit_action = QAction('&Exit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.setStatusTip('Exit Program')
        quit_action.triggered.connect(self.disconnectAndQuit)

        # Add actions to the file menu
        file_menu.addAction(quit_action)

        # =============== PLOT MENU =============== 
        output_menu = menu_bar.addMenu('&Output')

        plot_ripple_history = output_menu.addAction('Ripple &History')
        plot_ripple_history.setStatusTip('Plot history of Sharp-Wave ripples in the current session.')
        plot_ripple_history.triggered.connect(self.plotRippleHistory)

        plot_brain_atlas = output_menu.addAction('Brain &Atlas')
        plot_brain_atlas.setStatusTip('Show tetrode position in Brain Atlas')
        plot_brain_atlas.triggered.connect(self.showBrainAtlas)
        plot_brain_atlas.setShortcut('Ctrl+A')

        print_ripple_stats = output_menu.addAction('&Ripple Statistics')
        print_ripple_stats.setStatusTip('Show ripple stats')
        print_ripple_stats.triggered.connect(self.showRippleStats)

        # =============== SERIAL MENU =============== 
        serial_menu = menu_bar.addMenu('&Serial')
        select_serial_port = serial_menu.addAction('&Select Port')
        select_serial_port.setStatusTip('Select serial port to trigger stimulation on.')
        select_serial_port.triggered.connect(self.selectSerialPort)

        self.enable_serial_port = QAction('&Enable Port', self, checkable=True)
        self.enable_serial_port.setStatusTip('Enable default serial port.')
        self.enable_serial_port.triggered.connect(self.enableSerialPort)
        self.enable_serial_port.setChecked(DEFAULT_SERIAL_ENABLED)
        serial_menu.addAction(self.enable_serial_port)

        test_single_pulse = serial_menu.addAction('Test &Single')
        test_single_pulse.setStatusTip('Send single biphasic pulse on serial port.')
        test_single_pulse.triggered.connect(self.testSingleSerialPulse)

        test_continuous_pulse = serial_menu.addAction('Test &Continuous')
        test_continuous_pulse.setStatusTip('Send a stream of biphasic pulses on the serial port.')
        test_continuous_pulse.triggered.connect(self.testContinuousSerialPulse)

        # =============== STIM MENU =============== 
        stimulation_menu = menu_bar.addMenu('&Stimulation')
        stim_mode_manual = stimulation_menu.addAction('&Manual')
        stim_mode_manual.setStatusTip('Set stimulation mode to manual.')
        stim_mode_manual.triggered.connect(self.manualStimTrigger)
        stim_mode_manual.setShortcut('Ctrl+M')

        self.stim_mode_position = QAction('&Position', self, checkable=True)
        self.stim_mode_position.setStatusTip('Use position data (Trigger Zones) to simulate.')
        self.stim_mode_position.setChecked(DEFAULT_STIM_MODE_POSITION_ENABLED)
        self.stim_mode_position.triggered.connect(self.positionStimTrigger)

        self.stim_mode_velocity = QAction('&Velocity', self, checkable=True)
        self.stim_mode_velocity.setStatusTip('Use velocity threshold to simulate.')
        self.stim_mode_velocity.setChecked(DEFAULT_STIM_MODE_VELOCITY_ENABLED)
        self.stim_mode_velocity.triggered.connect(self.velocityStimTrigger)

        self.stim_mode_ripple = QAction('&Ripple', self, checkable=True)
        self.stim_mode_ripple.setStatusTip('Stimulate on Sharp-Wave Ripples.')
        self.stim_mode_ripple.setChecked(DEFAULT_STIM_MODE_RIPPLE_ENABLED)
        self.stim_mode_ripple.triggered.connect(self.rippleStimTrigger)

        stimulation_menu.addAction(self.stim_mode_position)
        stimulation_menu.addAction(self.stim_mode_ripple)

        # =============== PREFERENCES MENU =============== 
        preferences_menu = menu_bar.addMenu('&Preferences')
        ripple_preferences_menu = preferences_menu.addAction('&SWR Detection')
        ripple_preferences_menu.setStatusTip('Set SWR preferences.')
        ripple_preferences_menu.triggered.connect(self.collectSWRPreferences)

        self.update_ripple_stats_menu = QAction('&Update stats', self, checkable=True)
        self.update_ripple_stats_menu.setStatusTip('Toggle continuous update of ripple statistics.')
        self.update_ripple_stats_menu.setChecked(DEFAULT_UPDATE_RIPPLE_STATS)
        self.update_ripple_stats_menu.triggered.connect(self.setUpdateRippleStats)

        reset_ripple_stats_menu = preferences_menu.addAction('&Reset stats')
        reset_ripple_stats_menu.setStatusTip('Reset Sharp-Wave Ripple statistics.')
        reset_ripple_stats_menu.triggered.connect(self.resetSWRStats)

        spike_zoom_menu = preferences_menu.addMenu('Zoom')
        spike_zoom_in_action = spike_zoom_menu.addAction('Zoom In')
        spike_zoom_in_action.setStatusTip('Zoom in to the cluster plot')
        spike_zoom_in_action.setShortcut('Ctrl+=')
        spike_zoom_in_action.triggered.connect(self.zoomInClustering)

        spike_zoom_out_action = spike_zoom_menu.addAction('Zoom Out')
        spike_zoom_out_action.setStatusTip('Zoom out of the cluster plot')
        spike_zoom_out_action.setShortcut('Ctrl+-')
        spike_zoom_out_action.triggered.connect(self.zoomOutClustering)

        # Add remaning menu items to the menu
        preferences_menu.addAction(self.update_ripple_stats_menu)

    def getProcessingArgs(self):
        assert(len(PROCESSING_ARGS) == len(D_USER_PROCESSING_CHOICES))
        user_choices = QtHelperUtils.CheckBoxWidget(PROCESSING_ARGS, message="Select processing options.",\
                default_choices=list(D_USER_PROCESSING_CHOICES.values())).exec_()

        # Create a user dictionary with all the options
        self.user_processing_choices = D_USER_PROCESSING_CHOICES.copy()
        if user_choices[0] == QDialog.Accepted:
            for key_idx, key_val in enumerate(self.user_processing_choices):
                if key_idx in user_choices[1]:
                    self.user_processing_choices[key_val] = True
                else:
                    self.user_processing_choices[key_val] = False

    def setupActiveThreads(self):
        try:
            # First check if we are working with a linear track - This will
            # change how spikes, place fields and decoding are processed.
            # TODO: Can also use this to select trigger zones

            if self.user_processing_choices['linear']:
                print(MODULE_IDENTIFIER + "Switching analysis to Linear Track")
                self._is_linear_track = True

            # if self.user_processing_choices['lfp'] or self.user_processing_choices['calib']:
            if self.user_processing_choices['lfp']:
                print(MODULE_IDENTIFIER + "Starting LFP data Threads.")
                # LFP Threads
                self.initLFPThreads()

            if self.user_processing_choices['position']:
                # Position data is binned differently for linear track and open
                # field environments.
                print(MODULE_IDENTIFIER + "Starting Position data Threads.")
                if __debug__:
                    if self._is_linear_track:
                        self.position_estimator = PositionAnalysis.PositionEstimator(self.sg_client, \
                                n_bins=PositionAnalysis.N_LINEAR_TRACK_BINS,is_linear_track=True, \
                                write_position_log=True)
                    else:
                        self.position_estimator = PositionAnalysis.PositionEstimator(self.sg_client, \
                                is_linear_track=False, write_position_log=True)
                else:
                    if self._is_linear_track:
                        self.position_estimator = PositionAnalysis.PositionEstimator(self.sg_client, \
                                n_bins=PositionAnalysis.N_LINEAR_TRACK_BINS,is_linear_track=True)
                    else:
                        self.position_estimator = PositionAnalysis.PositionEstimator(self.sg_client, \
                                is_linear_track=False)
                self.active_processes.append(self.position_estimator)

            self.spike_sorter = self.spike_listener
            if self.user_processing_choices['lfp']:
                # Ripple triggered actions
                # This has to done after the threads above because this thread
                # write to the calibration plot thread after it has detected a
                # ripple.
                print(MODULE_IDENTIFIER + "Starting Sharp-Wave Ripple processing Threads.")
                self.initRippleTriggerThreads()

            if self.user_processing_choices['stim']:
                # Stimulation Threads
                print(MODULE_IDENTIFIER + "Starting Stimulation Threads.")
                self.initStimulationThreads()

        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + "Unable to start processing threads.")
            print(err)

    def connectSpikeGadgets(self):
        """
        Connect to Trodes client.
        """
        if self.sg_client is not None:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Already connected to Trodes! Try restarting to re-connect.')
            return

        try:
            self.sg_client = TrodesInterface.SGClient("ReplayInterruption")
        except Exception as err:
            print(err)
            self.sg_client = None
            user_response = QtHelperUtils.display_information(\
                    'Unable to connect to Trodes! Would you like to continue?')

            # Ask the user if they would like to continue
            if user_response == QMessageBox.Cancel:
                return


        # Use preferences to selectively start the desired threads
        self.getProcessingArgs()

        if self.sg_client is not None:
            self.statusBar().showMessage('Connected to SpikeGadgets. Press Ctrl+T to stream.')
        else:
            self.statusBar().showMessage('Not connected to SpikeGadgets.')

    def setupGraphicsPipelines(self):
        """
        Once all the threads have been set up, we can set up the graphics pipelines.
        In case of online clustering, this might take a little bit of pre-processing time.
        """
        try:
            # Start the graphical pipleline with this.
            self.graphical_interface = Visualization.GraphicsManager(\
                    self.spike_sorter, self.position_estimator, self.place_field_handler, \
                    self.stimulation_trigger, self.show_swr_trigger, self.show_sde_trigger, \
                    self.calib_trigger, self.decoding_done, self.user_processing_choices, \
                    is_linear_track=self._is_linear_track)

            # Load the tetrode and cluster information into the respective menus.
            self.graphical_interface.setTetrodeList(self.cluster_identity_map.keys(), \
                    load_adjusting_log=self.user_processing_choices['adjusting'])
        except Exception as err:
            print(err)
            return

        # Pause for a second here so that things can render
        time.sleep(1)

    def loadClusterFile(self, cluster_filename=None):
        """
        Load cluster information from a cluster file.
        """
        # Uncomment to use a hardcoded file
        # cluster_filename = "./config/Billy3_20181218.trodesClusters"
        # cluster_filename = "./config/Billy3_20181219_005635_merge_time_0.trodesClusters"
        # cluster_filename = "./config/full_config20190206_session_start.trodesClusters"
        # cluster_filename = "./config/full_config20190208_session_start.trodesClusters"
        # cluster_filename = "./config/full_config20190206_session_end.trodesClusters"
        # cluster_filename = "./config/full_config20190307.trodesClusters"
        # cluster_filename = "open_field_full_config20190220_172702.trodesClusters"

        # Set the configuration option above to choose a cluster file. We don't
        # really use hand-clustered cells anymore so commenting this out for
        # now. This need not be done if we have requested online-clustering
        if DEFAULT_CLUSTER_SELECTION_CHOICE:
            cluster_config = Configuration.read_cluster_file(cluster_filename, self.tetrodes_of_interest)
        else:
            cluster_config = None

        if cluster_config is not None:
            self.n_units = cluster_config[0]
            self.cluster_identity_map = cluster_config[1]
            if (self.n_units == 0):
                print(MODULE_IDENTIFIER + 'WARNING: No clusters found in the cluster file.')
            if self.tetrodes_of_interest is None:
                self.tetrodes_of_interest = list(self.cluster_identity_map.keys())
        else:
            logging.info(MODULE_IDENTIFIER + "Did not read cluster file. Using default map [64 tetrodes].")
            print("Using default cluster map.")
            self.n_units = 64
            self.cluster_identity_map = dict()
            self.tetrodes_of_interest = list()
            for tet_idx in range(1,65):
                self.cluster_identity_map[tet_idx] = {0: tet_idx-1}
                self.tetrodes_of_interest.append(tet_idx)

        # if __debug__:
        #     QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Read cluster identity map.')
        print(MODULE_IDENTIFIER + "Cluster Identity map...")
        print(self.cluster_identity_map)

        # NOTE: Using all the tetrodes that have clusters marked on them for ripple analysis
        self.n_tetrodes = len(self.cluster_identity_map)

    def initSingleUnitList(self):
        # Get a list of all the units that we need to listen to from Trodes
        self.session_unit_list = list()
        if not self.cluster_identity_map:
            self.loadClusterFile()
            for tetrode in self.cluster_identity_map.keys():
                tetrode_units = self.cluster_identity_map[tetrode].values()
                self.session_unit_list.extend(tetrode_units)

    def streamData(self):
        if self.data_streaming:
            self.stopThreads()
            self.data_streaming = False
        else:
            self.statusBar().showMessage('Streaming...')
            time.sleep(1)

            # Check we already have active processes in the pipeline
            if not self.active_processes:
                self.initSingleUnitList()
                self.setupActiveThreads()
            self.setupGraphicsPipelines()
            self.startActiveThreads()
            self.data_streaming = True
            # Get the geometry from the graphics manager
            layout_geometry = self.graphical_interface.getGeometry()
            self.setGeometry(100, 100, layout_geometry[0], layout_geometry[1])
            self.setCentralWidget(self.graphical_interface.widget)
            self.statusBar().showMessage('Analyzing data stream...')

            self.graphical_interface.init_dependent_vars(\
                    (self.shared_raw_lfp_buffer, self.shared_ripple_buffer), \
                    (self.shared_calib_plot_times, self.shared_calib_plot_counts),\
                    (self.shared_spk_times, self.shared_spk_rates), \
                    self.shared_place_fields, self.shared_posterior_buffer, \
                    self.shared_posterior_bin_times, clusters=self.session_unit_list)
            self.graphical_interface.start()
            self.startDelayedThreads()

    def freezeStream(self, state):
        """
        Continue streaming data but freeze the current display. Same function is used to toggle the freeze state.
        """
        if self.graphical_interface is None:
            QtHelperUtils.display_information('Graphics not initialized. Nothing to freeze -_-')
            self.freeze_action.setChecked(False)
            return

        self.graphical_interface.freeze(state)
        if state:
            self.statusBar().showMessage("Display frozen!")
        else:
            self.statusBar().showMessage("Display resumed!")

    def stopThreads(self):
        try:
            # Join all the threads to wait for their execution to  finish
            # Run cleanup here
            if __debug__:
                self.code_profiler.disable()
                self.code_profiler.dump_stats(self.profile_filename)

            for requested_process in self.active_processes:
                requested_process.join()

            with self._clustering_finished[1]:
                self._clustering_finished[0].clear()

            for requested_process in self.delayed_processes:
                requested_process.join()

            # After all the threads have been joined, delete them
            self.active_processes = list()
            self.delayed_processes = list()
        except Exception as err:
            logging.debug(MODULE_IDENTIFIER + "Caught Interrupt while exiting...")
            print(err)

        self.statusBar().showMessage('Streaming Paused!')

    def startActiveThreads(self):
        """
        Start all the active processing thread - These threads do not depend on
        others for initialization and can be started from the get go.
        """
        try:
            for requested_process in self.active_processes:
                requested_process.start()

        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to stream. Check connection to client!')
            print(err)
            return

        if __debug__:
            self.code_profiler.enable()

    def startDelayedThreads(self):
        """
        Start all the delayed threads. These threads cannot be started on their
        own. They need information from other (active) threads for their setup.
        """
        try:
            for requested_process in self.delayed_processes:
                requested_process.start()

        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to process stream. Check active processes!')
            print(err)
            return


    def initRippleTriggerThreads(self):
        """
        Initialize threads dependent on ripple triggers (these need pretty much everything!)
        """
        try:
            if self.shared_raw_lfp_buffer is None:
                self.shared_raw_lfp_buffer = RawArray(ctypes.c_double, self.n_tetrodes * RiD.LFP_BUFFER_LENGTH)

            if self.shared_ripple_buffer is None:
                self.shared_ripple_buffer = RawArray(ctypes.c_double, self.n_tetrodes * RiD.RIPPLE_POWER_BUFFER_LENGTH)

            self.ripple_detector = RippleAnalysis.RippleDetector(self.lfp_listener, self.calib_plot,\
                    trigger_condition=(self.swr_trig_condition, self.show_swr_trigger, self.calib_trigger),\
                    shared_buffers=(self.shared_raw_lfp_buffer, self.shared_ripple_buffer))
            self._ripple_preference_menu = QtHelperUtils.RippleSelectionMenuWidget(self.tetrodes_of_interest)
            self.active_processes.append(self.ripple_detector)

            self._ripple_preference_menu.setIdxs(self._ripple_reference_tet, self._ripple_baseline_tet)
            if self._ripple_reference_tet is not None:
                self.ripple_detector.set_ripple_reference(self._ripple_reference_tet)
            if self._ripple_baseline_tet is not None:
                self.ripple_detector.set_ripple_baseline(self._ripple_baseline_tet)

        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start ripple trigger threads(s).')
            print(err)
            return

    def initStimulationThreads(self):
        """
        Initialize threads for electrical/optical stimulation.
        """
        try:
            self.stimulation_trigger = StimulationSynchronizer(self.spike_listener,\
                    self.position_estimator, self.place_field_handler, self.swr_trig_condition, \
                    None, self.sg_client, self.serial_port)
            self.active_processes.append(self.stimulation_trigger)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start stimulation threads(s).')
            print(err)
            return


    def initLFPThreads(self):
        """
        Initialize the threads needed for LFP processing.
        """
        try:
            tetrode_argument = [str(tet) for tet in self.tetrodes_of_interest]
            self.lfp_listener = RippleAnalysis.LFPListener(self.sg_client, tetrode_argument)
            self.active_processes.append(self.lfp_listener)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start LFP thread(s).')
            print(err)
            return

def main():
    # Start logging before anything else
    log_file_prefix = "replay_disruption_log"
    log_filename = time.strftime(log_file_prefix + "_%Y%m%d_%H%M%S.log")
    if __debug__:
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.DEBUG, datefmt="%H:%M:%S")
    else:
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.INFO, datefmt="%H:%M:%S")
        """
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.DEBUG, datefmt="%H:%M:%S")
        """
    logging.info(MODULE_IDENTIFIER + "Starting Log file at " + time.ctime())

    qt_args = list()
    qt_args.append('OnlineInterruption.py')
    qt_args.append('-style')
    qt_args.append('Fusion')
    print(MODULE_IDENTIFIER + "Qt Arguments: " + str(qt_args))
    parent_app = QApplication(qt_args)
    print(MODULE_IDENTIFIER + "Parsing Input Arguments: " + str(sys.argv))
    parsed_arguments = QtHelperUtils.parseQtCommandlineArgs(sys.argv)
    command_window = CommandWindow(parsed_arguments)
    command_window.show()
    sys.exit(parent_app.exec_())

if (__name__ == "__main__"):
    main()
