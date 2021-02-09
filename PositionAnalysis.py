"""
Collect position data from trodes

TODO: Add a linearization routine. At the moment, we simply return the x-value.
"""
import csv
import threading
import time
from copy import copy
from datetime import datetime
import logging
import numpy as np
from multiprocessing import Pipe

# Local imports
import RippleDefinitions as RiD
import ThreadExtension

# Profiling
import cProfile

MODULE_IDENTIFIER = "[PositionAnalysis] "
DEFAULT_TRACK_GEOMETRY = [(2.5, 12), (11, 7.5)]
N_POSITION_BINS = (15, 16)
N_LINEAR_TRACK_BINS = 30
N_LINEAR_TRACK_YBINS = 16

# Define identifier for the left and right map that will be used everywhere
RIGHT_MAP = 0
LEFT_MAP  = 1

class PositionEstimator(ThreadExtension.StoppableThread):
    """
    Run a thread that collects position data from trodes.
    """

    # Min/Max position values in x and y to be used for binning
    # For Open Field
    """
    # For Open Field
    __P_MIN_X = -100
    __P_MIN_Y = -100
    __P_MAX_X = 1300
    __P_MAX_Y = 1100
    """

    # For linear track
    __P_MIN_X = 100
    __P_MIN_Y = 50
    __P_MAX_X = 1100
    __P_MAX_Y = 650

    """
    # For Krech Maze
    __P_MIN_X = 200
    __P_MIN_Y = 200
    __P_MAX_X = 1000
    __P_MAX_Y = 800
    """

    __P_BIN_SIZE_X = (__P_MAX_X - __P_MIN_X)
    __P_BIN_SIZE_Y = (__P_MAX_Y - __P_MIN_Y)
    __REAL_BIN_SIZE_X = RiD.FIELD_SIZE[0]/50.0
    __REAL_BIN_SIZE_Y = RiD.FIELD_SIZE[1]/50.0
    __SPEED_SMOOTHING_FACTOR = 0.75
    __MAX_TIMESTAMP_JUMP = 2000
    __MAX_REAL_TIME_JUMP = __MAX_TIMESTAMP_JUMP/RiD.SPIKE_SAMPLING_FREQ

    #def __init__(self, sg_client, n_bins, past_position_buffer, camera_number=1):
    def __init__(self, sg_client, n_bins=N_POSITION_BINS, camera_number=1, is_linear_track=False, \
            write_position_log=False):
        ThreadExtension.StoppableThread.__init__(self)
        self._data_field = np.ndarray([], dtype=[('timestamp', 'u4'), ('line_segment', 'i4'), \
                ('position_on_segment', 'f8'), ('position_x', 'i2'), ('position_y', 'i2')])
        # TODO: Take the camera number into account here. This could just be
        # the index of the camera window that is open and should be connected
        # to.
        self._position_consumer = sg_client.subscribeHighFreqData("PositionData", "CameraModule")
        self._is_linear_track = is_linear_track
        self._track_geometry = list()
        if self._is_linear_track:
            self._n_bins_x = int(n_bins)
            self._n_bins_y = N_LINEAR_TRACK_YBINS
        else:
            self._n_bins_x = int(n_bins[0])
            self._n_bins_y = int(n_bins[1])

        # self._bin_occupancy = np.zeros((self._n_bins_x, self._n_bins_y), dtype='float')
        #self._past_position_buffer = past_position_buffer

        # TODO: This is assuming that the jump in timestamps will not
        # completely fill up the memory. If the bin size is small, we might end
        # up filling the whole memory. We need this to  get appropriate
        # position bins for spikes in case the threads reading position and
        # spikes are not synchronized.
        if (self._position_consumer is None):
            # Failed to open connection to camera module
            logging.warning("Failed to open Camera Module")
            raise Exception("Error: Could not connect to camera, aborting.")
        self._position_consumer.initialize()
        self._csv_writer = None
        if write_position_log:
            csv_filename = time.strftime("position_data_log" + "_%Y%m%d_%H%M%S.csv")
            try:
                self._csv_file = open(csv_filename, mode='w')
                self._csv_writer = csv.writer(self._csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                self._csv_writer.writerow(['TIMESTAMP', 'POS_X', 'POS_Y', 'POS_LIN', 'VEL'])
            except Exception as err:
                logging.critical(MODULE_IDENTIFIER + "Unable to open log file.")
                print(err)

        self._position_buffer_connections = []
        logging.info(MODULE_IDENTIFIER + "Starting Position tracking thread")

    def getPositionBin(self):
        """
        Get the BIN for the current position.
        """

        # The position binning is different between linear track and a 2D
        # environement - In a linear environment, X is GOD, X is everything.
        x_bin, y_bin = self.getXYBin()

        # TODO Might have to do something more involved here
        #   - Find  the nearest point on the lineraized track.
        #   - Return the corresponding position bin.
        if self._is_linear_track:
            return x_bin

        return x_bin * self._n_bins_y + y_bin

    def get_linearized_position(self, x_pos, y_pos):
        """
        Use the track geometry to convert x and y position to a linearized
        track position.
        """

        return x_pos

    def getXYBin(self):
        """
        Get the x and y BIN for the current position.
        """
        px = self._data_field['position_x']
        py = self._data_field['position_y']

        # Camera data coming in has flipped Y-coordinates!
        # Instead of discretizing the position here, leave it as is. It can be
        # taken care of when building place fields. This will allow for much
        # better visualization.
        # x_bin = np.floor_divide(self._n_bins_x * (px - self.__P_MIN_X),self.__P_BIN_SIZE_X)
        # y_bin = np.floor_divide(self._n_bins_y * (self.__P_MAX_Y - py),self.__P_BIN_SIZE_Y)
        x_bin = np.divide(self._n_bins_x * (px - self.__P_MIN_X),self.__P_BIN_SIZE_X)
        y_bin = np.divide(self._n_bins_y * (self.__P_MAX_Y - py),self.__P_BIN_SIZE_Y)

        if x_bin < 0:
            x_bin = 0
        elif x_bin > self._n_bins_x-1:
            x_bin = self._n_bins_x-1

        if y_bin < 0:
            y_bin = 0
        elif y_bin > self._n_bins_y-1:
            y_bin = self._n_bins_y-1
        return (x_bin, y_bin)

    """
    def get_bin_occupancy(self):
        return np.copy(self._bin_occupancy)
    """

    def get_position_buffer_connection(self):
        my_end, your_end = Pipe()
        self._position_buffer_connections.append(my_end)
        return your_end

    def run(self):
        """
        Collect position data continuously and update the amount of time spent
        in each position bin
        """
        # Create and run profiler
        if __debug__:
            code_profiler = cProfile.Profile()
            profile_prefix = "position_fetcher_profile"
            profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")
            code_profiler.enable()

        # Keep track of current and previous BIN ID, and also the time at which last jump happened
        curr_x_bin = -1
        curr_y_bin = -1
        prev_x_bin = -1
        prev_y_bin = -1
        last_velocity = 0

        # TODO: Because it will not be possible to get the correct first time
        # stamp, we will have to ignore the first data entry obtained here.
        # Otherwise it will skew the occupancy!
        down_time = 0.0
        prev_step_timestamp = 0
        real_time_spent_in_prev_bin = 0.0
        real_distance_moved = 0.0
        linearized_position = 0.0

        # NOTE: For this thread, data is not streaming in quite as fast and as
        # a result, most of the time is spent in self.req_stop(). Maybe adding
        # a sleep to this will help.
        while not self.req_stop():
            n_available_frames = self._position_consumer.available(0)
            if n_available_frames == 0:
                down_time += 0.02
                time.sleep(0.02)
                if down_time > 1.0:
                    down_time = 0.0
                    print(MODULE_IDENTIFIER + "Warning: Not receiving position data.")
            else:
                down_time = 0.0

            for _ in range(n_available_frames):
                self._position_consumer.readData(self._data_field)
                current_timestamp = self._data_field['timestamp']
                (floating_x_bin, floating_y_bin) = self.getXYBin()
                linearized_position = self.get_linearized_position(floating_x_bin, floating_y_bin)
                curr_x_bin = int(np.round(floating_x_bin))
                curr_y_bin = int(np.round(floating_y_bin))
                if (prev_x_bin < 0):
                    try:
                        if __debug__:
                            print(MODULE_IDENTIFIER + 'Writing output buffers [1]...')
                        for outp in self._position_buffer_connections:
                            outp.send((current_timestamp, floating_x_bin, floating_y_bin, linearized_position, 0.0))
                        if __debug__:
                            print(MODULE_IDENTIFIER + 'Buffers [1] written...')
                    except BrokenPipeError as err:
                        print(MODULE_IDENTIFIER + 'Unable to write to Pipe. Aborting.')
                        print(err)
                        print(self._position_buffer_connections)
                        break

                    prev_x_bin = curr_x_bin
                    prev_y_bin = curr_y_bin
                    logging.info(MODULE_IDENTIFIER + "Position started (%d, %d, TS: %d)"%(curr_x_bin, curr_y_bin, current_timestamp))
                    prev_step_timestamp = copy(current_timestamp)
                elif ((curr_x_bin != prev_x_bin) or (curr_y_bin != prev_y_bin)):
                    time_spent_in_prev_bin = current_timestamp - prev_step_timestamp

                    # This is some serious overkill.. Most of the times, we
                    # will be moving by just 1 position bin... That too either
                    # in X or Y
                    real_time_spent_in_prev_bin = float(time_spent_in_prev_bin)/RiD.SPIKE_SAMPLING_FREQ

                    # The distance moved will have a sign accompanying it on linear track 
                    if self._is_linear_track:
                        real_distance_moved = self.__REAL_BIN_SIZE_X * (curr_x_bin-prev_x_bin)
                    else:
                        real_distance_moved = self.__REAL_BIN_SIZE_X * np.sqrt(pow(curr_x_bin-prev_x_bin,2) + \
                                self.__REAL_BIN_SIZE_Y * pow(curr_y_bin-prev_y_bin,2))
                    logging.debug(MODULE_IDENTIFIER + "Moved %.2fcm in %.2fs."%(real_distance_moved,real_time_spent_in_prev_bin))

                    # TODO: Add sign to the speed when working with a linear track
                    if (time_spent_in_prev_bin != 0):
                        last_velocity = (1 - self.__SPEED_SMOOTHING_FACTOR) * real_distance_moved/real_time_spent_in_prev_bin + \
                                self.__SPEED_SMOOTHING_FACTOR * last_velocity

                    for outp in self._position_buffer_connections:
                        outp.send((current_timestamp, floating_x_bin, floating_y_bin, linearized_position, last_velocity))

                    # Update the total time spent in the bin we were previously in
                    # self._bin_occupancy[prev_x_bin, prev_y_bin] += real_time_spent_in_prev_bin
                    # print(np.max(self._bin_occupancy))

                    # DEBUG: Report the jump in position bins
                    logging.debug(MODULE_IDENTIFIER + "Position jumped (%d, %d) -> (%d,%d), TS:%d"%(prev_x_bin, prev_y_bin, curr_x_bin, curr_y_bin, current_timestamp))
                    # logging.debug(MODULE_IDENTIFIER + "Position binned (%d, %d) = (%d,%d)"%(curr_x_bin, curr_y_bin, \
                    #       self._data_field['position_x'], self._data_field['position_y']))

                    # Update the current bin and timestamps
                    # An assignment here just binds the variable
                    # prev_step_timestamp to current_timestamp, never giving us
                    # the actual time  jump... Mystery
                    prev_step_timestamp = copy(current_timestamp)
                    prev_x_bin = curr_x_bin
                    prev_y_bin = curr_y_bin
                elif (current_timestamp - prev_step_timestamp) > self.__MAX_TIMESTAMP_JUMP:
                    # We know the animal is in the same position as before!
                    # TODO: Make speed half of its
                    real_time_spent_in_prev_bin += self.__MAX_REAL_TIME_JUMP
                    last_velocity = (1 - self.__SPEED_SMOOTHING_FACTOR) * real_distance_moved/real_time_spent_in_prev_bin + \
                            self.__SPEED_SMOOTHING_FACTOR * last_velocity
                    for outp in self._position_buffer_connections:
                        outp.send((current_timestamp, floating_x_bin, floating_y_bin, linearized_position, last_velocity))

                if self._csv_writer:
                    self._csv_writer.writerow([current_timestamp, floating_x_bin, floating_y_bin, linearized_position, last_velocity])

        if self._csv_writer:
            self._csv_file.close()

        if __debug__:
            code_profiler.disable()
            code_profiler.dump_stats(profile_filename)
        logging.info(MODULE_IDENTIFIER + "Position data collection Stopped")
