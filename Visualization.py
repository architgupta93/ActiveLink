"""
Visualization of various measures
List of TODO itemS:
- Add a trace of the past position data that can be cleared from the user menu
- Show spikes from forward and backward running directions on a linear track in different colors
"""

from collections import deque
from multiprocessing import Process, Event, Value, Lock
import tkinter
import time
import math
import copy
import random
import numpy as np
import colorsys
from scipy import signal
from scipy.stats.stats import pearsonr
from datetime import datetime
import threading
from scipy.ndimage.filters import gaussian_filter
import logging

# Matplotlib in Qt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.animation as animation

# Creating windows using PyQt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QPushButton, QSlider, QRadioButton, QLabel, QInputDialog, QTextEdit, QLineEdit, QCheckBox
from PyQt5.QtWidgets import QDialogButtonBox, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# Local Imports
import BrainAtlas
import AdjustingLog
import QtHelperUtils
import Configuration
import RippleAnalysis
import PositionAnalysis
import RippleDefinitions as RiD

USER_MESSAGE_IDENTIFIER = "[UserMessage] "
MODULE_IDENTIFIER = "[GraphicsHandler] "
VERIFY_BRAIN_COORDINATES = False
FORCE_IMAGE_DISPLAY = False
ANIMATION_INTERVAL = 20
MAX_N_CLUSTERED_SPIKES = 20000
LFP_TPTS = np.linspace(-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME, RiD.LFP_BUFFER_LENGTH)
SDE_TPTS = np.linspace(-0.5 * RiD.SDE_BUFFER_TIME, 0.5 * RiD.SDE_BUFFER_TIME, RiD.SDE_BUFFER_LENGTH)
RIPPLE_POWER_TPTS = np.linspace(-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME, RiD.RIPPLE_POWER_BUFFER_LENGTH)
SPK_COUNT_TPTS = np.linspace(0, RiD.CALIB_PLOT_BUFFER_TIME, RiD.CALIB_PLOT_BUFFER_LENGTH)
PEAK_LFP_AMPLITUDE = 1000
LFP_PLOT_OFFSET = 3.0
N_RIPPLES_TO_SHOW = 100

N_CHANNEL_MAPS = 6
CHANNEL_MAP_TEXT = [\
        "Ch 1 vs. Ch 2",\
        "Ch 1 vs. Ch 3",\
        "Ch 1 vs. Ch 4",\
        "Ch 2 vs. Ch 3",\
        "Ch 2 vs. Ch 4",\
        "Ch 3 vs. Ch 4"]
CHANNEL_MAP = [\
        [0, 1],\
        [0, 2],\
        [0, 3],\
        [1, 2],\
        [1, 3],\
        [2, 3]]
COLOR_SEED = random.random()

def generate_color_spread(n_colors):
    HSV_tuples = [(x*1.0/n_colors, 0.75, 0.75) for x in range(n_colors)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples] 
    return RGB_tuples

def generate_random_colors(n_colors):
    RGB_tuples = []
    r_val = int(COLOR_SEED * 256)
    g_val = int(COLOR_SEED * 256)
    b_val = int(COLOR_SEED * 256)
    step = 256 / n_colors
    for i in range(n_colors):
        r_val += step
        g_val += step
        b_val += step

        r_val = int(r_val) % 256
        g_val = int(g_val) % 256
        b_val = int(b_val) % 256
        RGB_tuples.append((r_val/256.0,g_val/256.0,b_val/256.0)) 
    return RGB_tuples

def normalizeData(in_data):
    # TODO: Might need tiling of data if there are multiple dimensions
    data_mean = np.mean(in_data, axis=0)
    data_std  = np.std(in_data, axis=0)
    norm_data = np.divide((in_data - data_mean), data_std)
    return (norm_data, data_mean, data_std)

def animateLFP(timestamps, lfp, raw_ripple, ripple_power, frame_size, statistic=None):
    """
    Animate a given LFP by plotting a fixed size sliding frame.

    :timestamps: time-points for the LFP data
    :lfp: LFP (Raw/Filtered) for a single tetrode
    :raw_ripple: LFP Passed through a ripple filter
    :ripple_power: Ripple power calculated over a moving winow centered at all
        the data points.
    :frame_size: Size of the frame that should be seen at once
    :statistic: Function handle that should be applied to the data to generate
        a scalar quantity that can also be plotted!
    """

    # Turn interactive plotting off. It messes up animation
    plt.ioff()

    # Change this to '3d' if the need every arises for a multi-dimensional plot
    lfp_fig   = plt.figure()
    plot_axes = plt.axes(projection=None)

    # Start with an empty plot, it can be then updated by animation functions
    # NOTE: The way frame is accessed in animation internals forces us to
    # make this an array if nothing else is being passed in. Having text
    # removes this requirement.
    lfp_frame,   = plot_axes.plot([], [], animated=True)
    r_raw_frame, = plot_axes.plot([], [], animated=True)
    r_pow_frame, = plot_axes.plot([], [], animated=True)
    txt_template = 't = %.2fs'
    lfp_measure  = plot_axes.text(0.5, 0.09, '', transform=plot_axes.transAxes)

    # Local functions for setting up animation frames and cycling through them
    def _nextAnimFrame(step=0):
        """
        # Making sure that the step index and data are coming in properly
        print(step)
        print(lfp[step])
        """
        lfp_frame.set_data(timestamps[step:step+frame_size], lfp[step:step+frame_size])
        r_raw_frame.set_data(timestamps[step:step+frame_size], raw_ripple[step:step+frame_size])
        r_pow_frame.set_data(timestamps[step:step+frame_size], ripple_power[step:step+frame_size])
        lfp_measure.set_text(txt_template % timestamps[step])
        # Updating the limits is needed still so that the correct range of data
        # is displayed! It doesn't update the axis labels though - That's a
        # different ballgame!
        plot_axes.set_xlim(timestamps[step], timestamps[step+frame_size])
        return lfp_frame, r_raw_frame, r_pow_frame, lfp_measure

    def _initAnimFrame():
        # NOTE: Init function called twice! I have seen this before but still
        # don't understand why it works this way!
        # print("Initializing animation frame...")
        plot_axes.set_xlabel('Time (s)')
        plot_axes.set_ylabel('EEG (uV)')
        plot_axes.set_ylim(min(lfp), max(lfp))
        plot_axes.set_xlim(timestamps[0], timestamps[frame_size])
        plot_axes.grid(True)
        return _nextAnimFrame()

    n_frames = len(timestamps) - frame_size
    lfp_anim = animation.FuncAnimation(lfp_fig, _nextAnimFrame, np.arange(0, n_frames), \
            init_func=_initAnimFrame, interval=RiD.LFP_ANIMATION_INTERVAL, \
            blit=True, repeat=False)
    plt.figure(lfp_fig.number)

    # Make the filtered ripple thinner
    r_raw_frame.set_linewidth(0.5)
    plt.show(plot_axes)

def visualizeLFP(timestamps, raw_lfp_buffer, ripple_power, ripple_filtered_lfp, \
        ripple_events=None, do_animation=False):
    # Normalize both EEG and Ripple power so that they can be visualized together.
    norm_lfp, lfp_mean, lfp_std = normalizeData(raw_lfp_buffer[0,:])
    norm_ripple_power, power_mean, power_std = normalizeData(ripple_power[0,:])
    norm_raw_ripple, ripple_mean, ripple_std = normalizeData(ripple_filtered_lfp[0,:])

    print("Ripple Statistics...")
    print("Mean: %.2f"%power_mean)
    print("Std: %.2f"%power_std)

    # Plots - Pick a tetrode to plot the data from
    # Static plots
    plt.ion()
    n_tetrodes = 1
    for tetrode_idx in range(n_tetrodes):
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(timestamps, norm_lfp)
        if ripple_events is not None:
            ax1.scatter(ripple_events, np.zeros(len(ripple_events)), c="r")
        ax1.grid(True)
        ax2.plot(timestamps, norm_raw_ripple)
        ax2.plot(timestamps, norm_ripple_power)
        ax2.grid(True)
        plt.show()

    """
    # Plot a histogram of the LFP power
    plt.figure()
    hist_axes = plt.axes()
    plt.hist(norm_ripple_power, bins=RiD.N_POWER_BINS, density=True)
    plt.grid(True)
    hist_axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.show()
    """

    if do_animation:
        # Animation
        wait_for_user_input = input('Press ENTER to continue, or Q to ABORT.')
        if (wait_for_user_input == 'Q'):
            return
        animateLFP(timestamps, norm_lfp, norm_raw_ripple, norm_ripple_power, 400)
    return (power_mean, power_std)

class RippleVisualizer(QDialog):
    """
    Class for displaying past ripples in a single window.
    """

    __N_DISPLAY_PANELS = 16
    __TNUM_XLEVEL = -0.24
    __TNUM_YLEVEL = 6.5

    __SWR_STAT_X_LEVEL = -0.24
    __SWR_STAT_Y_LEVEL = 5.2 

    __SWR_CORR_X_LEVEL = -0.03
    __SWR_CORR_Y_LEVEL = 6.5 

    __POW_STAT_X_LEVEL = 0.07
    __POW_STAT_Y_LEVEL = 6.5 

    __POW_CORR_X_LEVEL = 0.17
    __POW_CORR_Y_LEVEL = 6.5 

    def __init__(self, n_tetrodes, lfp_buffer, ripple_power_buffer, reference_tet=None):
        """
        Class constructor
        This class plots ripples on all the electrodes simultaneously, allowing
        you to compare the relative strength of SWR at different recording
        sites. It can also be a good method to select a candidate recording
        site for detecting Sharp-Wave ripples.
        """

        # Call parent class constructor
        QDialog.__init__(self)

        # Create an empty canvas to add the plots in
        self.figure  = Figure(figsize=(20,20))
        self.canvas  = FigureCanvas(self.figure)
        self.n_tetrodes = n_tetrodes

        self.swr_reference = reference_tet
        self.average_ripple_power = np.zeros(self.n_tetrodes)
        self.average_sharp_wave_amplitude = np.zeros(self.n_tetrodes)
        self.ripple_power_corr = np.zeros(self.n_tetrodes)
        self.sharp_wave_corr = np.zeros(self.n_tetrodes)
        self.sharp_wave = deque(maxlen=N_RIPPLES_TO_SHOW)

        # Track different ripple properties for each of the tetrodes

        n_grid_rows = np.ceil(np.sqrt(self.__N_DISPLAY_PANELS))
        n_grid_cols = np.ceil(self.__N_DISPLAY_PANELS/n_grid_rows)
        plot_grid = gridspec.GridSpec(int(n_grid_rows), int(n_grid_cols))

        # Keep track of the number of ripple we have been provided
        if len(lfp_buffer) != len(ripple_power_buffer):
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + \
                    "Unable to instantiate ripple view. Mismatch in raw LFP and power.")
            return

        self.lfp_buffer = lfp_buffer.copy()
        self.ripple_power_buffer = ripple_power_buffer.copy()
        self.n_ripples = len(self.lfp_buffer)
        self._current_ripple_idx = self.n_ripples-1
        self.figure.suptitle('Ripple %d of %d'%(self.n_ripples,self.n_ripples))

        # Set up display panels groups of recording data to be shown together
        self.n_panels = int(np.ceil(self.n_tetrodes/self.__N_DISPLAY_PANELS))
        panel_names = [str(x+1) + ': T' + str(x * self.__N_DISPLAY_PANELS + 1) \
                + '-T' + str((x+1) * self.__N_DISPLAY_PANELS) \
                for x in range(self.n_panels)]
        self.panel_selection = QComboBox()
        self.panel_selection.addItems(panel_names)
        self.panel_selection.currentIndexChanged.connect(self.plot_ripple_data)

        self._plot_axes = list()
        self._lfp_frames = list()
        self._tetrode_num_frame = list()
        self._sharp_wave_frames = list()
        self._sharp_wave_corr_frame = list()
        self._sharp_wave_stat_frame = list()
        self._ripple_power_stat_frame = list()
        self._ripple_power_corr_frame = list()
        self._power_frames = list()

        for grid_place in range(self.__N_DISPLAY_PANELS):
            tetrode_ax = self.figure.add_subplot(plot_grid[grid_place])
            tetrode_ax.set_xticks([])
            tetrode_ax.set_yticks([])
            tetrode_ax.set_xticklabels([])
            tetrode_ax.set_yticklabels([])
            tetrode_ax.set_xlim([LFP_TPTS[0], LFP_TPTS[-1]])
            tetrode_ax.set_ylim((-1.0, 1.6*RiD.RIPPLE_POWER_THRESHOLD))

            self._tetrode_num_frame.append(tetrode_ax.text(self.__TNUM_XLEVEL, self.__TNUM_YLEVEL, \
                    ''))
            if len(lfp_buffer) > 0:
                tetrode_lfp_frame, = tetrode_ax.plot(LFP_TPTS, LFP_PLOT_OFFSET + self.lfp_buffer[\
                    -1][grid_place,:]/PEAK_LFP_AMPLITUDE)
                tetrode_power_frame, = tetrode_ax.plot(RIPPLE_POWER_TPTS, self.ripple_power_buffer[\
                    -1][grid_place,:])
            else:
                tetrode_lfp_frame, = tetrode_ax.plot([], [])
                tetrode_power_frame, = tetrode_ax.plot([], [])

            tetrode_sharp_wave_frame, = tetrode_ax.plot([], [])
            tetrode_sharp_wave_corr_frame = tetrode_ax.text(self.__SWR_CORR_X_LEVEL, \
                    self.__SWR_CORR_Y_LEVEL, '')
            tetrode_sharp_wave_stat_frame = tetrode_ax.text(self.__SWR_STAT_X_LEVEL, \
                    self.__SWR_STAT_Y_LEVEL, '')
            tetrode_ripple_power_stat_frame = tetrode_ax.text(self.__POW_STAT_X_LEVEL, \
                    self.__POW_STAT_Y_LEVEL, '')
            tetrode_ripple_power_corr_frame = tetrode_ax.text(self.__POW_CORR_X_LEVEL, \
                    self.__POW_CORR_Y_LEVEL, '')

            # Save the axes objects as well as the draw frames.
            self._plot_axes.append(tetrode_ax)
            self._lfp_frames.append(tetrode_lfp_frame)
            self._power_frames.append(tetrode_power_frame)
            self._sharp_wave_frames.append(tetrode_sharp_wave_frame)
            self._sharp_wave_corr_frame.append(tetrode_sharp_wave_corr_frame)
            self._sharp_wave_stat_frame.append(tetrode_sharp_wave_stat_frame)
            self._ripple_power_stat_frame.append(tetrode_ripple_power_stat_frame)
            self._ripple_power_corr_frame.append(tetrode_ripple_power_corr_frame)

        # Add some basic buttons
        self.button_box = QDialogButtonBox(Qt.Horizontal)
        self.next_button = QPushButton('Next')
        self.prev_button = QPushButton('Prev')
        self.save_button = QPushButton('Save')
        self.close_button = QPushButton('Close')
        self.button_box.addButton(self.next_button, QDialogButtonBox.ActionRole)
        self.button_box.addButton(self.prev_button, QDialogButtonBox.ActionRole)
        self.button_box.addButton(self.save_button, QDialogButtonBox.ActionRole)
        self.button_box.addButton(self.close_button, QDialogButtonBox.RejectRole)

        # Attach functions to the individual buttons
        self.next_button.clicked.connect(self.show_next_swr)
        self.prev_button.clicked.connect(self.show_prev_swr)
        self.save_button.clicked.connect(self.save_swr)
        self.close_button.clicked.connect(self.reject)

        # Set up the layout for the dialog box
        g_layout = QVBoxLayout()
        g_layout.addWidget(self.panel_selection)
        g_layout.addWidget(self.canvas)
        g_layout.addWidget(self.button_box, alignment=Qt.AlignCenter)
        self.setLayout(g_layout)
        self.extract_swr_metrics()
        self.plot_ripple_data()

    def extract_swr_metrics(self):
        """
        Extract numerical metrics for all the sharp-wave ripples in the buffer.
        """

        nyq_freq = RiD.LFP_FREQUENCY * 0.5
        lo_cutoff = RiD.SHARPWAVE_LO_FREQ/nyq_freq
        hi_cutoff = RiD.SHARPWAVE_HI_FREQ/nyq_freq
        pl, ph = signal.butter(RiD.SHARPWAVE_FILTER_ORDER, [lo_cutoff, hi_cutoff], btype='band')

        # Go through the data ripple by ripple and calculate statistics.
        for r_idx in range(self.n_ripples):
            # Extract the sharp-wave envelope first.
            this_ripple_sharp_wave = signal.filtfilt(pl, ph, self.lfp_buffer[r_idx], axis=1)
            self.sharp_wave.append(this_ripple_sharp_wave)

            # Average ripple power
            self.average_ripple_power += np.mean(self.ripple_power_buffer[r_idx], axis=1)
            self.average_sharp_wave_amplitude += np.mean(abs(this_ripple_sharp_wave), axis=1)

            if self.swr_reference is not None:
                r_power_corr = np.empty(self.n_tetrodes)
                r_sharpwave_corr = np.empty(self.n_tetrodes)

                # TODO: We can make use of the p-values in making this data as
                # well. However, I suspect, there to be a significant
                # correlation for most of the data.
                for t_idx in range(self.n_tetrodes):
                    # Ripple power correlation
                    r_power_corr[t_idx], _ = pearsonr(self.ripple_power_buffer[r_idx][t_idx,:],\
                            self.ripple_power_buffer[r_idx][self.swr_reference,:])

                    # Sharp-wave correlation
                    r_sharpwave_corr[t_idx], _ = pearsonr(this_ripple_sharp_wave[t_idx,:],\
                            this_ripple_sharp_wave[self.swr_reference,:])
                self.ripple_power_corr += r_power_corr
                self.sharp_wave_corr += r_sharpwave_corr 

        if self.n_ripples > 0:
            # Divide all the measures by the number of ripples we have so far.
            self.average_sharp_wave_amplitude /= self.n_ripples
            self.average_ripple_power /= self.n_ripples
            self.ripple_power_corr /= self.n_ripples
            self.sharp_wave_corr /= self.n_ripples

    def plot_ripple_data(self):
        """
        Show data in the currently seleceted panel.
        """

        if self.n_ripples <= 0:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + "No ripples to display.")
            return

        current_panel_selection = self.panel_selection.currentIndex()
        self.figure.suptitle('Ripple %d of %d'%(1+self._current_ripple_idx,self.n_ripples))
        # It is possible that we can an empty buffer. Barring that possibility,
        # we should be ok here - Plotting the most recent ripple.
        for t_idx, tetrode_ax in enumerate(self._plot_axes):
            # Get the tetrode number and specify it in the panel
            grid_place = current_panel_selection * self.__N_DISPLAY_PANELS + t_idx
            self._tetrode_num_frame[t_idx].set_text('T%d'%(grid_place+1))

            self._lfp_frames[t_idx].set_data(LFP_TPTS, LFP_PLOT_OFFSET + self.lfp_buffer[\
                    self._current_ripple_idx][grid_place,:]/PEAK_LFP_AMPLITUDE)
            self._power_frames[t_idx].set_data(RIPPLE_POWER_TPTS, self.ripple_power_buffer[\
                    self._current_ripple_idx][grid_place,:])

            # Now plotting the quantities that are derived by this thread.
            """
            # 3 line plots is way too many to be clearly understandable.
            self._sharp_wave_frames[t_idx].set_data(LFP_TPTS, LFP_PLOT_OFFSET + self.sharp_wave[\
                    self._current_ripple_idx][grid_place,:]/PEAK_LFP_AMPLITUDE)
            """
            self._sharp_wave_corr_frame[t_idx].set_text(round(self.sharp_wave_corr[grid_place], 2))
            self._sharp_wave_stat_frame[t_idx].set_text(round(self.average_sharp_wave_amplitude[grid_place], 2))
            self._ripple_power_corr_frame[t_idx].set_text(round(self.ripple_power_corr[grid_place], 2))
            self._ripple_power_stat_frame[t_idx].set_text(round(self.average_ripple_power[grid_place], 2))

            # Plot the extracted sharp-wave
        self.canvas.draw()

    def show_next_swr(self):
        """
        Show the next ripple in buffer (if there is one).
        """
        self._current_ripple_idx = min(self.n_ripples-1, self._current_ripple_idx+1);
        self.plot_ripple_data()

    def show_prev_swr(self):
        """
        Show the previous ripple in buffer (if there is one).
        """
        self._current_ripple_idx = max(0, self._current_ripple_idx-1);
        self.plot_ripple_data()

    def save_swr(self):
        """
        Save both the current image as well as the stats.
        """

        self.save_ripple_stats()
        self.save_ripple_plots()

    def save_ripple_stats(self):
        """
        Save the statistics generated from post processing of the detected sharp-wave ripples.
        """
        save_file_name = time.strftime("swr_post_processes_stats_%Y%m%d_%H%M%S.npz") 
        try:
            np.savez(save_file_name, self.average_ripple_power, self.ripple_power_corr, \
                    self.average_sharp_wave_amplitude, self.sharp_wave_corr)
        except Exception as err:
            print(MODULE_IDENTIFIER + "Unable to save current display.")
            print(err)

    def save_ripple_plots(self):
        """
        Save all the ripples in the buffer as different images.
        """

        # Create a filename
        save_file_name = time.strftime("Panel" + str(self.panel_selection.currentIndex()) + \
                "_swr_history_%Y%m%d_%H%M%S.png") 
        try:
            self.figure.savefig(save_file_name)
        except Exception as err:
            print(MODULE_IDENTIFIER + "Unable to save current display.")
            print(err)


class GraphicsManager(Process):
    """
    Process for managing visualization and graphics
    """

    __N_POSITION_ELEMENTS_TO_PLOT = 100
    __N_SPIKES_TO_PLOT = 2000
    __N_ANIMATION_FRAMES = 50000
    __PLACE_FIELD_REFRESH_RATE = 1
    __PLOT_REFRESH_RATE = 0.05
    __CLUSTERS_TO_PLOT = []
    __N_SUBPLOT_COLS = int(3)
    __MAX_FIRING_RATE = 40.0
    __POSTERIOR_TIMEOUT = 0.1
    __JUMP_DISTANCE = 100
    __FREEZE_SLEEP_TIME = 1.0
    __RIPPLE_DETECTION_TIMEOUT = 0.1
    __RIPPLE_FETCH_SLEEP_TIME = 0.01
    __SDE_DETECTION_TIMEOUT = 0.1
    __SDE_FETCH_SLEEP_TIME = 0.01
    __RIPPLE_SMOOTHING_WINDOW = 2
    __DECODED_SMOOTHING_COM_FACTOR = 0.1
    __DECODED_SMOOTHING_POSTERIOR_FACTOR = 0.01
    __MIN_DISPLAY_POSTERIOR = 0.05
    __POSTERIOR_SMOOTHING_WINDOW = 2
    __POSTERIOR_SCALING_FACTOR = 2.0

    def __init__(self, spike_clustering, position_estimator, place_field_handler, \
            stimulation_trigger_thread, swr_trigger_condition, sde_trigger_condition, \
            calib_trigger_condition, decoding_condition, processing_args, \
            is_linear_track=False):
        """
        Graphical Manager for all the processes
        :spike_clustering: Thread listening to incoming raw spike stream from trodes.
        :position_estimator: Thread listening to position data from camera stream
        :place_field_handler: Process constructing place fields
        :swr_trigger_condition: Trigger condition for sharp-wave ripples
        :sde_trigger_condition: Trigger condition for spike-density events
        """
        Process.__init__(self)

        # Graphics windows
        self.widget  = QDialog()
        self.figure  = Figure(figsize=(12,16))
        self.canvas  = FigureCanvas(self.figure)
        self.show_clustering = False
        self._is_linear_track = is_linear_track
        self._show_decoded_posterior = True
        self._processing_args = processing_args

        # Count the number of requested features
        # UPDATE: Instead of checking data fields, we are explicitly fetching
        # processing arguments from the use and using that to determine what
        # will be displayed. Keeping the old checks around for now.
        n_features_to_show = 0

        if self._processing_args['lfp']:
        # if ripple_buffers[0] is not None:
            # Ripple data needs to be shown
            n_features_to_show += 1

        self.show_clustering = False
        if self._processing_args['position']:
        # if position_estimator is not None:
            n_features_to_show += 1

        # The layout of the application can be different based on what features
        # are being requested
        # print(n_features_to_show)
        if n_features_to_show == 0:
            self._geometry = (700, 250)
        elif n_features_to_show == 1:
            plot_grid = gridspec.GridSpec(1, 1)
            self._geometry = (700, 750)
        elif n_features_to_show == 2:
            plot_grid = gridspec.GridSpec(1, 2)
            self._geometry = (900, 600)
        elif n_features_to_show == 3:
            plot_grid = gridspec.GridSpec(1, 3)
            self._geometry = (1400, 600)
        elif n_features_to_show == 4:
            plot_grid = gridspec.GridSpec(2, 2)
            self._geometry = (900, 950)
        elif n_features_to_show == 5:
            plot_grid = gridspec.GridSpec(3, 2)
            self._geometry = (900, 950)
        else:
            self._geometry = (700, 750)

        self.toolbar = NavigationToolbar(self.canvas, self.widget)
        self._display_lock = Lock()
        self._update_display = Value("b", True)

        # The checks have to be done here once again because we need to
        # generate the plot grid first. Only then can we follow up with
        # addition of these graphs into the figure window. 
        current_grid_place = 0
        if self._processing_args['lfp']:
        # if ripple_buffers[0] is not None:
            self._rd_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            current_grid_place += 1
        else:
            self._rd_ax = None

        self._cp_ax = None

        if self.show_clustering:
            self._cluster_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            self.channel_map_selection = QComboBox()
            self.channel_map_selection.addItems(CHANNEL_MAP_TEXT)
            current_grid_place += 1
            self._cluster_amplitude = 600.0
        else:
            self._cluster_ax = None
            self.channel_map_selection = None
            self._cluster_amplitude = 0.0

        if self._processing_args['position']:
        # if position_estimator is not None:
            self._spk_pos_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            current_grid_place += 1
        else:
            self._spk_pos_ax = None

        self._sde_ax = None
        self._pf_ax = None
        self._dec_ax = None

        # Selecting individual units
        self.unit_selection = QComboBox()
        self.user_message = QTextEdit()
        self.user_message.resize(300, 100)
        self.adjusting_dist = QLineEdit()
        self.cortical_activity_checkbox = QCheckBox("Cortex")
        self.white_matter_checkbox = QCheckBox("White Matter")
        self.sharp_wave_ripple_checkbox = QCheckBox("SWR")
        self.hippocampal_cells_checkbox = QCheckBox("HPC")
        self.log_message = QPushButton('Log')
        self.clear_message = QPushButton('Clear')
        self.log_message.clicked.connect(self.LogUserMessage)
        self.clear_message.clicked.connect(self.ClearUserMessage)

        # Brain Atlas interface
        self.brain_atlas = BrainAtlas.WebAtlas()
        self.adjusting_log = None

        # self.unit_selection.currentIndexChanged.connect(self.refresh)
        # Add next and prev buttons to look at individual cells.
        self.next_unit_button = QPushButton('Next')
        self.next_unit_button.clicked.connect(self.NextUnit)
        self.prev_unit_button = QPushButton('Prev')
        self.prev_unit_button.clicked.connect(self.PrevUnit)

        # Selecting individual tetrodes
        self.tetrode_selection = QComboBox()
        self.tetrode_selection.currentIndexChanged.connect(self.ClearUserMessage)

        # Add next and prev buttons to look at individual cells.
        self.next_tet_button = QPushButton('Next')
        self.next_tet_button.clicked.connect(self.NextTetrode)
        self.prev_tet_button = QPushButton('Prev')
        self.prev_tet_button.clicked.connect(self.PrevTetrode)

        self._keep_running = Event()
        self._spike_clustering = spike_clustering
        self._position_estimator = position_estimator
        self._decoding_condition = decoding_condition
        self._place_field_handler = place_field_handler
        self._stim_trigger_thread = stimulation_trigger_thread
        self._swr_trigger_condition = swr_trigger_condition[0]
        self._swr_data_access = swr_trigger_condition[1]
        self._sde_trigger_condition = sde_trigger_condition[0]
        self._sde_data_access = sde_trigger_condition[1]
        self._calib_trigger_condition = calib_trigger_condition[0]
        self._calib_data_access = calib_trigger_condition[1]

        # Get and plot the location of the stimulation zones.
        if self._stim_trigger_thread is not None:
            self._trigger_zones = self._stim_trigger_thread.get_trigger_zones()
            self._n_trigger_zones = len(self._trigger_zones)
            if __debug__:
                print(self._trigger_zones)
        else:
            self._trigger_zones = None
            self._n_trigger_zones = 0

        # Create a list of threads depending on the requeseted features.
        self._thread_list = list()

        # Enable clustering handlers if requested
        self._cluster_lock = Lock()
        self._do_color_clusters = True
        self._n_tetrodes = 0
        if self.show_clustering:
            # Get a handle to the collected spike waveform peaks.
            self._cluster_buffer = self._spike_clustering.get_spike_peak_connection()
            self._n_tetrodes = self._spike_clustering.get_n_tetrodes()

            # Make a list of clustered spikes for each tetrode and then for any
            # particular tetrode, maintain adeque of the 
            self._clustered_spikes = list()
            self._cluster_identities = list()
            self._unique_clusters = list()
            for t_idx in range(self._n_tetrodes):
                self._clustered_spikes.append(deque(maxlen=MAX_N_CLUSTERED_SPIKES))
                self._cluster_identities.append(deque(maxlen=MAX_N_CLUSTERED_SPIKES))
                self._unique_clusters.append(dict())

            self._thread_list.append(threading.Thread(name="ClusterFetcher", daemon=True, \
                    target=self.fetch_clustered_spikes))
            logging.info(MODULE_IDENTIFIER + "Added spike-clustering threads to graphics pipeline.")
        else:
            self._cluster_buffer = None
            self._clustered_spikes = None
            self._cluster_identities = None
            self._unique_clusters = None


        # Figure/Animation element. So far the following have been included
        # Ripple detection
        # Place Fields
        # Position/Spikes overalaid
        self._rd_frame = list()
        self._cluster_frame = list()
        self._spk_pos_frame = list()
        self._sde_frame = list()
        self._pf_frame = list()
        self._dec_frame = list()
        self._cp_frame = list()
        self._anim_objs = list()

        logging.info(MODULE_IDENTIFIER + "Graphics interface started.")
        self.setLayout()
        self.clearAxes()

    def init_dependent_vars(self, ripple_buffers, calib_plot_buffers, sde_buffers, \
            shared_place_fields, shared_posterior, shared_posterior_bin_times,
            clusters=None):
        """
        Setup dependent variables for the graphical interface.
        :clusters: User specified cluster indices that we should be looking at.
        """
        if (self._spike_clustering is not None):
            self._n_total_clusters = self._spike_clustering.get_n_clusters()
            session_unit_list = [x for x in range(self._n_total_clusters)]
            self.setUnitList(session_unit_list)
        else:
            self._n_total_clusters = 0
            if clusters is None:
                self._n_clusters = len(self.__CLUSTERS_TO_PLOT)
                self._clusters = self.__CLUSTERS_TO_PLOT
            else:
                self._clusters = clusters
                self._n_clusters = len(self._clusters)

        self._cluster_colormap = colormap.magma(np.linspace(0, 1, self._n_clusters))

        # Enable Ripple Buffer and corresponding thread if requested
        if self._processing_args['lfp']:
        # if ripple_buffers[0] is not None:
            # If we are not getting spike data, then this needs to be updated using the LFP BUFFER
            if self._n_tetrodes == 0:
                self._n_tetrodes = int(len(ripple_buffers[0])/RiD.LFP_BUFFER_LENGTH)
            self._shared_raw_lfp_buffer = np.reshape(np.frombuffer(ripple_buffers[0], dtype='double'), \
                    (self._n_tetrodes, RiD.LFP_BUFFER_LENGTH))
            self._shared_ripple_power_buffer = np.reshape(np.frombuffer(ripple_buffers[1], dtype='double'), \
                    (self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH))
            self._thread_list.append(threading.Thread(name="RippleFrameFetcher", daemon=True, \
                    target=self.fetch_incident_ripple))
            logging.info(MODULE_IDENTIFIER + "Added Ripple threads to Graphics pipeline.")
        else:
            self._shared_raw_lfp_buffer = None
            self._shared_ripple_power_buffer = None

        # Enable position data and thread if requested
        if self._processing_args['position']:
        # if self._position_estimator is not None:
            self._position_buffer = self._position_estimator.get_position_buffer_connection()
            self._thread_list.append(threading.Thread(name="PositionFetcher", daemon=True, \
                    target=self.fetch_position_and_update_frames))
            logging.info(MODULE_IDENTIFIER + "Added Position threads to Graphics pipeline.")

        # Enable spike density thread if requested
        self._shared_sde_spk_rates = None
        self._shared_sde_spk_times = None

        # Enable spike calibration plots if requested
        self._calib_lock = threading.Lock()
        self._mean_spike_suppression = 0.0
        self._n_calibration_frames = 0
        self._shared_calib_plot_times = None
        self._shared_calib_plot_counts = None

        ############################################################
        ############## DELAYED PROCESSING THREADS ##################
        ############################################################
        # Enable place field handler if requested
        self._spike_buffer = None
        self._shared_place_fields = None

        # Enable posterior fetcher thread if requested.
        self._decoding_lock = threading.Lock()
        self.dec_CoM = None
        self.peak_posterior = None
        self.raw_CoM_grid = None
        self._shared_posterior = None
        self._shared_posterior_bin_times = None

        # Local copies of the shared data that can be used at a leisurely pace
        self._lfp_lock = threading.Lock()
        self._lfp_tpts = LFP_TPTS
        self._sde_tpts = SDE_TPTS
        self._ripple_power_tpts = RIPPLE_POWER_TPTS

        # Buffers for visualizing all tetrodes simultaneously
        self._lfp_history_buffer = deque(maxlen=N_RIPPLES_TO_SHOW)
        self._ripple_power_history_buffer = deque(maxlen=N_RIPPLES_TO_SHOW)

        # Local LFP and ripple power buffers
        self._local_lfp_buffer = np.zeros((self._n_tetrodes, RiD.LFP_BUFFER_LENGTH), dtype='double')
        self._local_ripple_power_buffer = np.zeros((self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH), dtype='double')
        self._pf_lock = threading.Lock()
        if self._is_linear_track:
            self._most_recent_pf = np.zeros((PositionAnalysis.N_LINEAR_TRACK_BINS, \
                    2), dtype='float')
        else:
            self._most_recent_pf = np.zeros((PositionAnalysis.N_POSITION_BINS[0], \
                    PositionAnalysis.N_POSITION_BINS[1]), dtype='float')

        # Calibration plot local arrays
        self._spk_cnt_tpts = SPK_COUNT_TPTS
        self._local_spk_times_buffer = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH), dtype='double')
        self._local_spk_count_buffer = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH), dtype='double')

        # Spike density plot local arrays
        self._sde_lock = threading.Lock()
        self._local_sde_spk_times_buffer = np.zeros((RiD.SDE_BUFFER_LENGTH), dtype='double')
        self._local_sde_spk_rates_buffer = np.zeros((RiD.SDE_BUFFER_LENGTH), dtype='double')

        # Automatically keep only a fixed number of entries in this buffer... Useful for plotting
        self._pos_timestamps = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_lock = threading.Lock()
        self._pos_x = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_y = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._speed = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)

        # Maintain a separate deque for each cluster to plot
        self._spike_lock = threading.Lock()
        self.initSpikeDeque()

        # Start the animation for Spike-Position figure, place field figure
        self.initialize_ripple_detection_fig()
        self.initialize_spike_pos_fig()

        self.pauseAnimation()
        self._keep_running.set()
        for p__thread in self._thread_list:
            p__thread.start()
        print(MODULE_IDENTIFIER + 'Animation plots initialized.')

    def scale_cluster_zoom(self, scale_factor):
        self._cluster_amplitude *= scale_factor
        self._cluster_ax.set_xlim((0, self._cluster_amplitude))
        self._cluster_ax.set_ylim((0, self._cluster_amplitude))
        self.canvas.draw()

    def display_past_ripples(self, reference_tetrode):
        """
        Pull up a dialog box that shows the past ripples.
        """

        ripple_display = RippleVisualizer(self._n_tetrodes, \
                self._lfp_history_buffer, self._ripple_power_history_buffer,\
                reference_tetrode)
        ripple_display.exec_()


    def freeze(self, state):
        """
        Freeze/Unfreeze display update.
        """
        with self._display_lock:
            self._update_display.value = not state
            if state:
                logging.info(MODULE_IDENTIFIER + "Stream display frozen.")
            else:
                logging.info(MODULE_IDENTIFIER + "Stream display resumed.")

    def getGeometry(self):
        return self._geometry

    def initSpikeDeque(self):
        # In an open field, we only use the right map. On a linear track, the
        # spike colors identify which map they came from and as a result, two
        # different deques are maintained for the two running directions.
        self._spk_pos_x_rmap = []
        self._spk_pos_y_rmap = []
        self._spk_pos_x_lmap = []
        self._spk_pos_y_lmap = []
        for cl_idx in range(self._n_clusters):
            self._spk_pos_x_rmap.append(deque([], self.__N_SPIKES_TO_PLOT))
            self._spk_pos_y_rmap.append(deque([], self.__N_SPIKES_TO_PLOT))
            self._spk_pos_x_lmap.append(deque([], self.__N_SPIKES_TO_PLOT))
            self._spk_pos_y_lmap.append(deque([], self.__N_SPIKES_TO_PLOT))

    def setLayout(self):
        parent_layout_box = QVBoxLayout()
        parent_layout_box.addWidget(self.toolbar)
        if self.show_clustering:
            parent_layout_box.addWidget(self.channel_map_selection)
        parent_layout_box.addWidget(self.canvas)
        parent_layout_box.addStretch(1)

        # Controls for looking at individual units
        vbox_unit_buttons = QVBoxLayout()
        unit_selection_label = QLabel('Cell ID', alignment=Qt.AlignCenter)
        vbox_unit_buttons.addWidget(unit_selection_label)
        vbox_unit_buttons.addWidget(self.unit_selection)
        vbox_unit_buttons.addWidget(self.next_unit_button)
        vbox_unit_buttons.addWidget(self.prev_unit_button)

        # Controls for looking at individual tetrodes for LFP
        vbox_tetrode_buttons = QVBoxLayout()
        tetrode_selection_label = QLabel('nTrode ID', alignment=Qt.AlignCenter)
        vbox_tetrode_buttons.addWidget(tetrode_selection_label)
        vbox_tetrode_buttons.addWidget(self.tetrode_selection)
        vbox_tetrode_buttons.addWidget(self.next_tet_button)
        vbox_tetrode_buttons.addWidget(self.prev_tet_button)

        # Add a block for user to add comments
        checkbox_selection = QVBoxLayout()
        checkbox_selection.addWidget(self.cortical_activity_checkbox)
        checkbox_selection.addWidget(self.white_matter_checkbox)
        checkbox_selection.addWidget(self.sharp_wave_ripple_checkbox)
        checkbox_selection.addWidget(self.hippocampal_cells_checkbox)

        message_button_box = QHBoxLayout()
        message_button_box.addStretch(1)
        message_button_box.addWidget(self.adjusting_dist)
        message_button_box.addWidget(self.log_message)
        message_button_box.addWidget(self.clear_message)
        message_button_box.addStretch(1)

        vbox_user_message = QVBoxLayout()
        vbox_user_message.addWidget(self.user_message)
        vbox_user_message.addStretch(1)
        vbox_user_message.addLayout(message_button_box)

        # Put the tetrode and unit buttons together
        hbox_unit_and_tet_controls = QHBoxLayout()
        hbox_unit_and_tet_controls.addLayout(checkbox_selection)
        hbox_unit_and_tet_controls.addLayout(vbox_user_message)
        hbox_unit_and_tet_controls.addLayout(vbox_unit_buttons)
        hbox_unit_and_tet_controls.addLayout(vbox_tetrode_buttons)

        parent_layout_box.addLayout(hbox_unit_and_tet_controls)
        QDialog.setLayout(self.widget, parent_layout_box)

    def setClusterIdentities(self, cluster_identity_map):
        # Take a cluster identity map and use it to populate the tetrodes and units.
        pass

    def setUnitList(self, unit_list):
        # Take the list of units and set them as the current list of units to be looked at.
        # print(unit_list)
        unit_id_strings = [str(unit_id) for unit_id in unit_list]
        self.unit_selection.addItems(unit_id_strings)

        # Update the cluster information
        self._n_clusters = len(unit_list)
        self._clusters = unit_list
        self.initSpikeDeque()

    def setTetrodeList(self, tetrode_list, load_adjusting_log):
        # Start a new adjusting log for this tetrode list.
        # TODO: Do the log stuff only if we are actively adjusting.
        if load_adjusting_log:
            self.adjusting_log = AdjustingLog.TetrodeLog(tetrode_list)
        tetrode_id_strings = [str(tet_id) for tet_id in tetrode_list]
        self.tetrode_selection.addItems(tetrode_id_strings)

    def autosaveAdjustingLog(self):
        if self.adjusting_log is not None:
            logfile_name = time.strftime("AutoSave__AdjustingLog_%Y%m%d_%H%M%S.json") 
            self.adjusting_log.writeDataFile(logfile_name)
        else:
            QtHelperUtils.display_warning("Adjusting log not initialized.")

    def showTetrodeInBrain(self, force=False):
        if self.adjusting_log is not None:
            default_coordinates = self.adjusting_log.getCoordinates(self.tetrode_selection.currentText())
        else:
            default_coordinates = list()

        if force:
            if default_coordinates:
                self.brain_atlas.getCoronalImage(*default_coordinates)
            return

        user_response, coordinates, view_selection = QtHelperUtils.BrainCoordinateWidget(*default_coordinates).exec_()
        if user_response == QDialog.Accepted:
            if 0 in view_selection:
                self.brain_atlas.getCoronalImage(*coordinates)
            if 1 in view_selection:
                self.brain_atlas.getSagittalImage(*coordinates)
            if 2 in view_selection:
                self.brain_atlas.getHorizontalImage(*coordinates)

    def LogUserMessage(self):
        properties_tag = " -"
        tag_list = list()
        if self.cortical_activity_checkbox.isChecked():
            properties_tag += "C"
            tag_list.append('C')
        if self.white_matter_checkbox.isChecked():
            properties_tag += "W"
            tag_list.append('W')
        if self.sharp_wave_ripple_checkbox.isChecked():
            properties_tag += "S"
            tag_list.append('S')
        if self.hippocampal_cells_checkbox.isChecked():
            properties_tag += "H"
            tag_list.append('H')
        properties_tag += "-"
        tetrode_adjustment = self.adjusting_dist.text()
        # If no adjustment was made -- Let's say just a message has been logged, set the value to 0
        if not tetrode_adjustment:
            tetrode_adjustment = "0"

        user_text = self.user_message.toPlainText()
        logging.info(USER_MESSAGE_IDENTIFIER + user_text + " [Tags:" + \
                "T" + self.tetrode_selection.currentText() + properties_tag + "] " + \
                tetrode_adjustment)

        # If adjusting, update the tetrode's current position
        if self.adjusting_log is not None:
            self.adjusting_log.updateDepth(self.tetrode_selection.currentText(), \
                    float(tetrode_adjustment))
            self.adjusting_log.addTags(self.tetrode_selection.currentText(), \
                    tag_list)
            self.adjusting_log.addMessage(self.tetrode_selection.currentText(), \
                    time.strftime('[%Y.%m.%d %H:%M:%S] ' + properties_tag + " " + \
                    tetrode_adjustment + ' ' +  user_text))
        self.ClearUserMessage()

    def ClearUserMessage(self):
        if self.adjusting_log is not None:
            current_tags = self.adjusting_log.getTags(self.tetrode_selection.currentText())
            current_messages = self.adjusting_log.printMessages(self.tetrode_selection.currentText())
        else:
            current_tags = []

        if 'C' in current_tags:
            self.cortical_activity_checkbox.setChecked(True)
        else:
            self.cortical_activity_checkbox.setChecked(False)

        if 'W' in current_tags:
            self.white_matter_checkbox.setChecked(True)
        else:
            self.white_matter_checkbox.setChecked(False)

        if 'S' in current_tags:
            self.sharp_wave_ripple_checkbox.setChecked(True)
        else:
            self.sharp_wave_ripple_checkbox.setChecked(False)

        if 'H' in current_tags:
            self.hippocampal_cells_checkbox.setChecked(True)
        else:
            self.hippocampal_cells_checkbox.setChecked(False)

        self.user_message.setPlainText("")
        self.adjusting_dist.clear()
        if FORCE_IMAGE_DISPLAY:
            self.showTetrodeInBrain(force=VERIFY_BRAIN_COORDINATES)

    # Saving Images
    def saveDisplay(self):
        if self.figure is None:
            return

        # Create a filename
        save_file_name = time.strftime("T" + str(self.tetrode_selection.currentText()) + "_%Y%m%d_%H%M%S.png") 
        save_success = False
        try:
            self.figure.savefig(save_file_name)
            save_success = True
        except Exception as err:
            print(MODULE_IDENTIFIER + "Unable to save current display.")
            print(err)
        return save_success

    # Cluster viewing options
    def set_cluster_colors(self, state):
        self._do_color_clusters = not state

    def clear_clusters(self):
        if self.show_clustering:
            for t_idx in range(self._n_tetrodes):
                self._cluster_identities[t_idx].clear()
                self._clustered_spikes[t_idx].clear()

    # Saving Videos
    def recordDisplay(self):
        pass

    def NextUnit(self):
        current_unit = self.unit_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Unit: %d'%current_unit)
        if current_unit < self.unit_selection.count()-1:
            self.unit_selection.setCurrentIndex(current_unit+1)
        if self._is_linear_track:
            print(MODULE_IDENTIFIER + "Right: %d, Left: %d, spikes received for current unit"%\
                    (len(self._spk_pos_x_rmap[current_unit]), len(self._spk_pos_x_lmap[current_unit])))
        else:
            print(MODULE_IDENTIFIER + "%d spikes received for current unit"%len(self._spk_pos_x_rmap[current_unit]))

    def PrevUnit(self):
        current_unit = self.unit_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Unit: %d'%current_unit)
        if current_unit > 0:
            self.unit_selection.setCurrentIndex(current_unit-1)
        if self._is_linear_track:
            print(MODULE_IDENTIFIER + "Right: %d, Left: %d, spikes received for current unit"%\
                    (len(self._spk_pos_x_rmap[current_unit]), len(self._spk_pos_x_lmap[current_unit])))
        else:
            print(MODULE_IDENTIFIER + "%d spikes received for current unit"%len(self._spk_pos_x_rmap[current_unit]))

    def NextTetrode(self):
        current_tet = self.tetrode_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Tetrode: %d'%current_tet)
        with self._lfp_lock:
            if current_tet < self.tetrode_selection.count()-1:
                self.tetrode_selection.setCurrentIndex(current_tet+1)

    def PrevTetrode(self):
        current_tet = self.tetrode_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Tetrode: %d'%current_tet)
        with self._lfp_lock:
            if current_tet > 0:
                self.tetrode_selection.setCurrentIndex(current_tet-1)

    def clearAxes(self):
        # Ripple detection axis
        if self._rd_ax is not None:
            self._rd_ax.cla()
            self._rd_ax.set_xlabel("Time (s)")
            self._rd_ax.set_ylabel("Ripple Power (STD)")
            self._rd_ax.set_xlim((-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME))
            self._rd_ax.set_ylim((-1.0, 1.6*RiD.RIPPLE_POWER_THRESHOLD))
            self._rd_ax.grid(True)

        # Clustering plot
        if self._cluster_ax is not None:
            self._cluster_ax.cla()
            self._cluster_ax.set_xlabel('Amp (uV)')
            self._cluster_ax.set_ylabel('Amp (uV)')
            self._cluster_ax.set_xlim((0, self._cluster_amplitude))
            self._cluster_ax.set_ylim((0, self._cluster_amplitude))
            self._cluster_ax.grid(True)

        # Calibration plot
        if self._cp_ax is not None:
            self._cp_ax.cla()
            self._cp_ax.set_xlabel("Time (s)")
            self._cp_ax.set_ylabel("Spike Rate (spks/5ms)")
            self._cp_ax.set_xlim((-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME))
            self._cp_ax.set_ylim((0.0, 3.0))
            self._cp_ax.grid(True)

        # Spike density plot
        if self._sde_ax is not None:
            self._sde_ax.cla()
            self._sde_ax.set_xlabel("Time (s)")
            self._sde_ax.set_ylabel("Firing Rate (z-scored)")
            self._sde_ax.set_xlim((-0.5 * RiD.SDE_BUFFER_TIME, 0.5 * RiD.SDE_BUFFER_TIME))
            self._sde_ax.set_ylim((-1, 1.6 * RiD.SDE_ZSCORE_THRESHOLD))
            self._sde_ax.grid(True)

        # Place field
        if self._pf_ax is not None:
            self._pf_ax.cla()
            if self._is_linear_track:
                self._pf_ax.set_xlabel("Position (bin)")
                self._pf_ax.set_ylabel("Firing Rate (Hz)")
                self._pf_ax.set_xlim((0, PositionAnalysis.N_LINEAR_TRACK_BINS))
                self._pf_ax.set_ylim((0, self.__MAX_FIRING_RATE))
            else:
                self._pf_ax.set_xlabel("x (bin)")
                self._pf_ax.set_ylabel("y (bin)")
                self._pf_ax.set_xlim((0, -0.5+PositionAnalysis.N_POSITION_BINS[1]))
                self._pf_ax.set_ylim((0, -0.5+PositionAnalysis.N_POSITION_BINS[0]))
            self._pf_ax.grid(True)

        # Spikes (from a single cell) and position
        if self._spk_pos_ax is not None:
            self._spk_pos_ax.cla()
            self._spk_pos_ax.set_xlabel("x (bin)")
            self._spk_pos_ax.set_ylabel("y (bin)")
            if self._is_linear_track:
                self._spk_pos_ax.set_xticks([0, 10, 20, 30])
                self._spk_pos_ax.set_xlim((0, PositionAnalysis.N_LINEAR_TRACK_BINS))
                self._spk_pos_ax.set_ylim((0, PositionAnalysis.N_LINEAR_TRACK_YBINS))
            else:
                self._spk_pos_ax.set_xticks([0, 5, 10, 15])
                self._spk_pos_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
                self._spk_pos_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
            self._spk_pos_ax.set_yticks([0, 4, 8, 12, 16])
            self._spk_pos_ax.grid(False)

        if self._dec_ax is not None:
            self._dec_ax.cla()
            if self._is_linear_track:
                self._dec_ax.set_xlabel("Time Bin")
                self._dec_ax.set_ylabel("Position Bin")
            else:
                self._dec_ax.set_xlabel("x (bin)")
                self._dec_ax.set_ylabel("y (bin)")
                self._dec_ax.set_xticks([0, 5, 10, 15])
                self._dec_ax.set_yticks([0, 4, 8, 12, 16])
            self._dec_ax.grid(True)

        self.canvas.draw()

    def kill_gui(self):
        self._keep_running.clear()
        
    def update_cluster_frame(self, step=0):
        """
        Function used to show update spikes and their cluster allocations.
        """

        current_tetrode_selection = self.tetrode_selection.currentIndex()
        current_channel_selection = self.channel_map_selection.currentIndex()
        current_ch_ax1 = CHANNEL_MAP[current_channel_selection][0]
        current_ch_ax2 = CHANNEL_MAP[current_channel_selection][1]

        # TODO: Implement selecting one of the 6 channel combinations and
        # then displaying clustering in the peak amplitude space in that
        # view.
        with self._cluster_lock:
            tetrode_clusteterd_data = np.asarray(self._clustered_spikes[current_tetrode_selection])

            if len(tetrode_clusteterd_data) > 0:
                # TODO: This might not be the best way to do it. For now we will
                # generate a ton of colors and then just assign colors by cluster
                # idx. A lot of cluster idxs don't have any cells associated with
                # them but that is a difficulty for later.
                highest_cluster_idx = len(self._unique_clusters[current_tetrode_selection])

                # print(MODULE_IDENTIFIER + "Max cluster idx: %d"%highest_cluster_idx)
                # print(self._cluster_identities[current_tetrode_selection])
                cluster_colors = generate_color_spread(highest_cluster_idx+1)
                # cluster_colors = generate_random_colors(highest_cluster_idx+1)

                if self._do_color_clusters:
                    tetrode_cluster_colors = [cluster_colors[self._unique_clusters[current_tetrode_selection][x]] for x in self._cluster_identities[current_tetrode_selection]]
                else:
                    tetrode_cluster_colors = ['#1f77b4' for x in self._cluster_identities[current_tetrode_selection]]

                self._cluster_frame[0].set_offsets(np.concatenate((\
                    np.reshape(tetrode_clusteterd_data[:,current_ch_ax1], (-1,1)),\
                    np.reshape(tetrode_clusteterd_data[:,current_ch_ax2], (-1,1))), axis=1))
                self._cluster_frame[0].set_color(tetrode_cluster_colors)
            else:
                self._cluster_frame[0].set_offsets(np.zeros((0,2)))

        return self._cluster_frame

    def update_ripple_detection_frame(self, step=0):
        """
        Function used to show a ripple frame whenever a ripple is trigerred.
        This is a little different from the other frame update functions as it
        does not continuously update the frame but only when a ripple is triggerred.
        """

        # NOTE: This call blocks access to swr_trigger_condition for
        # __RIPPLE_DETECTION_TIMEOUT, which could be a long while. Don't let
        # this block any important functionality.
        with self._lfp_lock:
            current_tetrode_selection = self.tetrode_selection.currentIndex()
            self._rd_frame[0].set_data(self._lfp_tpts, LFP_PLOT_OFFSET + \
                    self._local_lfp_buffer[current_tetrode_selection,:]/PEAK_LFP_AMPLITUDE)

            # Smooth out the ripple power
            """
            smoothed_ripple_power = \
                    gaussian_filter(self._local_ripple_power_buffer[current_tetrode_selection,:], \
                    self.__RIPPLE_SMOOTHING_WINDOW)
            """
            smoothed_ripple_power = self._local_ripple_power_buffer[current_tetrode_selection,:]
            self._rd_frame[1].set_data(self._ripple_power_tpts, smoothed_ripple_power)
        return self._rd_frame

    def update_spike_density_frame(self, step=0):
        """
        Function used to show a spike density frame whenever a spike-density event is triggered.
        """

        with self._sde_lock:
            self._sde_frame[0].set_data(self._sde_tpts, \
                    self._local_sde_spk_rates_buffer)
        return self._sde_frame

    def update_calib_plot_frame(self, step=0):
        """
        Function used to show a ripple frame whenever a ripple is trigerred.
        This is a little different from the other frame update functions as it
        does not continuously update the frame but only when a ripple is triggerred.
        """

        # NOTE: This call blocks access to swr_trigger_condition for
        # __RIPPLE_DETECTION_TIMEOUT, which could be a long while. Don't let
        # this block any important functionality.
        with self._calib_lock:
            self._cp_frame[0].set_data(self._local_spk_times_buffer/RiD.SPIKE_SAMPLING_FREQ, \
                    self._local_spk_count_buffer)
            self._cp_frame[1].set_text("supp idx. " +str(round(self._mean_spike_suppression,2)))
        return self._cp_frame

    def update_position_and_spike_frame(self, step=0):
        """
        Function used for animating the current position of the animal.
        """

        cl_idx = max(self.unit_selection.currentIndex(), 0)
        with self._spike_lock:
            if self._n_clusters > 0:
                if self._is_linear_track:
                    self._spk_pos_frame[0].set_data((self._spk_pos_x_rmap[cl_idx], self._spk_pos_y_rmap[cl_idx]))
                    self._spk_pos_frame[1].set_data((self._spk_pos_x_lmap[cl_idx], self._spk_pos_y_lmap[cl_idx]))
                else:
                    self._spk_pos_frame[0].set_data((self._spk_pos_x_rmap[cl_idx], self._spk_pos_y_rmap[cl_idx]))

        with self._pos_lock:
            self._spk_pos_frame[-2].set_data((self._pos_x, self._pos_y))
            if len(self._speed) > 0:
                self._spk_pos_frame[-1].set_text('speed = %.2fcm/s'%self._speed[-1])
        return self._spk_pos_frame

    def update_place_field_frame(self, step=0):
        """
        Function used for animating the place field for a particular spike cluster.
        TODO: Utility to be expanded to multiple clusters in the future.

        :step: Animation iteration
        :returns: Animation frames to be plotted.
        """

        # print("Peak FR: %.2f, Mean FR: %.2f"%(np.max(self._most_recent_pf), np.mean(self._most_recent_pf)))
        # print("Min FR: %.2f, Max FR: %.2f"%(np.min(self._most_recent_pf), np.max(self._most_recent_pf)))
        # min_fr = np.min(self._most_recent_pf)
        # max_fr = np.max(self._most_recent_pf)
        with self._pf_lock:
            if self._is_linear_track:
                # TODO: Update the place field plot here
                self._pf_frame[0].set_ydata(self._most_recent_pf[:,PositionAnalysis.RIGHT_MAP])
                self._pf_frame[1].set_ydata(self._most_recent_pf[:,PositionAnalysis.LEFT_MAP])
            else:
                self._pf_frame[0].set_array(self._most_recent_pf.T)
        return self._pf_frame
        
    def fetch_incident_ripple(self):
        """
        Fetch raw LFP data and ripple power data.
        """
        logging.info(MODULE_IDENTIFIER + "Ripple frame pipe opened.")
        ripple_triggered = False
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                # Do not update the data frame! Keep everything as is.
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            with self._lfp_lock:
                self._swr_trigger_condition.wait(self.__RIPPLE_DETECTION_TIMEOUT)
                ripple_triggered = self._swr_trigger_condition.is_set()
                self._swr_trigger_condition.clear()
                if ripple_triggered:
                    with self._swr_data_access:
                        np.copyto(self._local_lfp_buffer, self._shared_raw_lfp_buffer)
                        np.copyto(self._local_ripple_power_buffer, self._shared_ripple_power_buffer)
                    logging.info(MODULE_IDENTIFIER + "Peak ripple power in frame %.2f"%np.max(self._local_ripple_power_buffer))

                    # Put the lfp data into the history buffer so that we can easily look at the last few ripples.
                    self._lfp_history_buffer.append(self._local_lfp_buffer.copy())
                    self._ripple_power_history_buffer.append(self._local_ripple_power_buffer.copy())

            time.sleep(self.__RIPPLE_FETCH_SLEEP_TIME)
        logging.info(MODULE_IDENTIFIER + "Ripple frame pipe closed.")

    def fetch_calibration_plot(self):
        """
        Fetch raw LFP data and ripple power data.
        """
        logging.info(MODULE_IDENTIFIER + "Calibration pipe opened.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                # Do not update the data frame! Keep everything as is.
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            with self._calib_lock:
                self._calib_trigger_condition.wait(self.__RIPPLE_DETECTION_TIMEOUT)
                calib_triggered = self._calib_trigger_condition.is_set()
                self._calib_trigger_condition.clear()
                if calib_triggered:
                    # print(MODULE_IDENTIFIER + "Calibration plot updated buffers.")
                    with self._calib_data_access:
                        np.copyto(self._local_spk_times_buffer, self._shared_calib_plot_times)
                        np.copyto(self._local_spk_count_buffer, self._shared_calib_plot_counts)
                    # print(self._local_spk_times_buffer)
                    # print(self._local_spk_count_buffer)

                    # Update the spike suppression measure
                    # Find 0 in the time-point array. This is the timestamp at
                    # which stimulation was triggered.
                    self._n_calibration_frames += 1
                    stim_t_idx = np.searchsorted(self._local_spk_times_buffer, 0)-1
                    # print(stim_t_idx)
                    if stim_t_idx > 0:
                        post_stim_spike_times = min(len(self._local_spk_count_buffer), stim_t_idx + RiD.CALIB_MEASURE_LENGTH)
                        pre_stim_spike_times = max(0, stim_t_idx - RiD.CALIB_MEASURE_LENGTH)
                        # print(post_stim_spike_times)
                        # print(pre_stim_spike_times)

                        pre_stim_spike_rate = np.mean(self._local_spk_count_buffer[pre_stim_spike_times:stim_t_idx])
                        post_stim_spike_rate = np.mean(self._local_spk_count_buffer[stim_t_idx:post_stim_spike_times])
                        post_pre_spike_rate_diff = (post_stim_spike_rate-pre_stim_spike_rate)/(pre_stim_spike_rate+post_stim_spike_rate)
                        self._mean_spike_suppression += post_pre_spike_rate_diff/self._n_calibration_frames

            time.sleep(self.__RIPPLE_FETCH_SLEEP_TIME)
        logging.info(MODULE_IDENTIFIER + "Calibration pipe closed.")

    def fetch_posterior_plot(self):
        """
        Fetch and plot the posterior.
        """
        logging.info(MODULE_IDENTIFIER + "Posterior pipe opened.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                # Do not update the data frame! Keep everything as is.
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            with self._decoding_lock:
                with self._decoding_condition:
                    decoding_finished = self._decoding_condition.wait(self.__POSTERIOR_TIMEOUT)
                    if decoding_finished:
                        np.copyto(self._local_posterior_buffer, self._shared_posterior)
                        np.copyto(self._local_posterior_bin_times, self._shared_posterior_bin_times)
            time.sleep(self.__POSTERIOR_TIMEOUT)
        logging.info(MODULE_IDENTIFIER + "Decoding pipe closed.")

    def fetch_place_fields(self):
        """
        Fetch place field data from place field handler.
        """
        logging.info(MODULE_IDENTIFIER + "Place Field pipe opened.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                # Do not update the data frame! Keep everything as is.
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            time.sleep(self.__PLACE_FIELD_REFRESH_RATE)
            # Request place field handler to pause place field calculation
            # while we fetch the data
            self._place_field_handler.submit_immediate_request()
            with self._pf_lock:
                # Uncomment this line to get an average of all the place fields
                # np.mean(self._shared_place_fields, out=self._most_recent_pf, axis=0)

                # Uncomment to look at the place field of the selected unit
                np.copyto(self._most_recent_pf, self._shared_place_fields[self.unit_selection.currentIndex(), :, :])
            # Release the request that paused place field computation
            self._place_field_handler.end_immediate_request()
        logging.info(MODULE_IDENTIFIER + "Place Field pipe closed.")

    def fetch_clustered_spikes(self):
        logging.info(MODULE_IDENTIFIER + "Clustering pipe openend.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            # We know that cluster_buffer can be polled because it was created
            # before this thread was assigned to a worker.
            if self._cluster_buffer.poll():
                [t_idx, cl_idx, spike_peaks] = self._cluster_buffer.recv()
                with self._cluster_lock:
                    self._clustered_spikes[t_idx].append(np.array(spike_peaks))
                    self._cluster_identities[t_idx].append(cl_idx)

                    # Check if the cluster is already known. If it hasn't, add an entry for it.
                    if cl_idx not in self._unique_clusters[t_idx]:
                        self._unique_clusters[t_idx][cl_idx] = len(self._unique_clusters[t_idx])
                        # print(MODULE_IDENTIFIER + "T%d New spike from cluster %d, assigned index %d"%(\
                        #         t_idx, cl_idx, self._unique_clusters[t_idx][cl_idx]))
                    logging.debug(MODULE_IDENTIFIER + "Fetched spike from tetrode %d, cluster: %d"%(\
                            t_idx, cl_idx))
            else:
                time.sleep(self.__PLOT_REFRESH_RATE)
        logging.info(MODULE_IDENTIFIER + "Clustering pipe closed.")

    def fetch_spikes_and_update_frames(self):
        logging.info(MODULE_IDENTIFIER + "Spike pipe opened.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                # Do not update the data frame! Keep everything as is.
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            if self._spike_buffer.poll():
                spike_data = self._spike_buffer.recv()
                # TODO: This is a little inefficient. For every spike we get,
                # we check to see if it is in the clusters of interest and then
                # find its  
                if spike_data[0] in self._clusters:
                    data_idx = self._clusters.index(spike_data[0])
                    with self._spike_lock:
                        if self._is_linear_track:
                            if spike_data[-1] > 0:
                                self._spk_pos_x_rmap[data_idx].append(spike_data[1])
                                self._spk_pos_y_rmap[data_idx].append(spike_data[2])
                            else:
                                self._spk_pos_x_lmap[data_idx].append(spike_data[1])
                                self._spk_pos_y_lmap[data_idx].append(spike_data[2])
                        else:
                            self._spk_pos_x_rmap[data_idx].append(spike_data[1])
                            self._spk_pos_y_rmap[data_idx].append(spike_data[2])
                logging.debug(MODULE_IDENTIFIER + "Fetched spike from cluster: %d, in bin (%d, %d). TS: %d, v=%.2fcm/s"%spike_data)
            else:
                time.sleep(self.__PLOT_REFRESH_RATE)
        logging.info(MODULE_IDENTIFIER + "Spike pipe closed.")

    def fetch_sde_events(self):
        logging.info(MODULE_IDENTIFIER + "SDE pipe opened.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            with self._sde_lock:
                self._sde_trigger_condition.wait(self.__SDE_DETECTION_TIMEOUT)
                sde_triggered = self._sde_trigger_condition.is_set()
                self._sde_trigger_condition.clear()
                if sde_triggered:
                    with self._sde_data_access:
                        np.copyto(self._local_sde_spk_rates_buffer, self._shared_sde_spk_rates)
                        np.copyto(self._local_sde_spk_times_buffer, self._shared_sde_spk_times)
                    logging.info(MODULE_IDENTIFIER + "Peak density in frame %.2f"%\
                            np.max(self._local_sde_spk_rates_buffer))

            time.sleep(self.__SDE_FETCH_SLEEP_TIME)
        logging.info(MODULE_IDENTIFIER + "Spike-density frame pipe closed.")

    def fetch_position_and_update_frames(self):
        logging.info(MODULE_IDENTIFIER + "Position pipe opened.")
        while self._keep_running.is_set():
            with self._display_lock:
                do_frame_update = self._update_display.value

            if not do_frame_update:
                # Do not update the data frame! Keep everything as is.
                time.sleep(self.__FREEZE_SLEEP_TIME)
                continue

            if self._position_buffer.poll():
                position_data = self._position_buffer.recv()
                with self._pos_lock:
                    self._pos_timestamps.append(position_data[0])
                    self._pos_x.append(position_data[1])
                    self._pos_y.append(position_data[2])
                    self._speed.append(position_data[-1])
                # print(self)
                # print(self._pos_x)
                # print(self._pos_y)
                logging.debug(MODULE_IDENTIFIER + "Fetched Position data... (%d, %d), v: %.2fcm/s"% \
                      (position_data[1],position_data[2], position_data[3]))
            else:
                time.sleep(self.__PLOT_REFRESH_RATE)
        logging.info(MODULE_IDENTIFIER + "Position pipe closed.")

    def pauseAnimation(self):
        """
        Pause all animation sources.
        """
        for ao in self._anim_objs:
            ao.event_source.stop()
    
    def playAnimation(self):
        """
        Play all animation sources.
        """
        for ao in self._anim_objs:
            ao.event_source.start()

    def initialize_ripple_detection_fig(self):
        """
        Initialize figure window for showing raw LFP and ripple power.
        :returns: TODO
        """
        if self._rd_ax is None:
            return

        lfp_frame, = self._rd_ax.plot([], [], animated=True)
        ripple_power_frame, = self._rd_ax.plot([], [], animated=True)
        self._rd_ax.legend((lfp_frame, ripple_power_frame), ('Raw LFP', 'Ripple Power'))
        self._rd_frame.append(lfp_frame)
        self._rd_frame.append(ripple_power_frame)

        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_ripple_detection_frame, frames=np.arange(self.__N_ANIMATION_FRAMES), \
                interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Ripple detection frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_cluster_plot_fig(self):
        """
        Initialize a figure window for showing clustered spike data (amplitude space).
        """
        if self._cluster_ax is None:
            return

        spk_cl_frame = self._cluster_ax.scatter([], [], s=2, \
                        alpha=0.9, animated=True)
        self._cluster_frame.append(spk_cl_frame)

        # Create animation object for showing the clustered spikes.
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_cluster_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), \
                interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Clustering frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_calib_plot_fig(self):
        """
        Initialize figure window for showing raw LFP and ripple power.
        :returns: TODO
        """
        if self._cp_ax is None:
            return

        spk_cnt_frame, = self._cp_ax.plot([], [], animated=True)
        suppression_measure_frame = self._cp_ax.text(0.0, 2.65, '', horizontalalignment='center', animated=True)
        self._cp_frame.append(spk_cnt_frame)
        self._cp_frame.append(suppression_measure_frame)

        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_calib_plot_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), \
                interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Spike calibration frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_sde_fig(self):
        """
        Initialize figure window for showing spike density events.
        """
        if self._sde_ax is None:
            return

        sde_frame, = self._sde_ax.plot([], [], animated=True)
        self._sde_frame.append(sde_frame)
        
        # TODO: Optionally, add some text teling what the timestamp for the spike density event was.
        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_spike_density_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), \
                interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Spike density frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_spike_pos_fig(self):
        """
        Initialize figure window for showing spikes overlaid on position
        """
        if self._spk_pos_ax is None:
            return

        spk_frame_right_map, = self._spk_pos_ax.plot([], [], linestyle='None', marker='o', \
                markersize=4, markerfacecolor='C0', markeredgecolor='C0', alpha=0.4, animated=True)
        spk_frame_left_map, = self._spk_pos_ax.plot([], [], linestyle='None', marker='o', \
                markersize=4, markerfacecolor='C3', markeredgecolor='C3', alpha=0.4, animated=True)
        pos_frame, = self._spk_pos_ax.plot([], [], color='C1', animated=True)
        vel_frame  = self._spk_pos_ax.text(0.5 * PositionAnalysis.N_POSITION_BINS[0], \
                0.02 * PositionAnalysis.N_POSITION_BINS[1], 'speed = 0cm/s')
        self._spk_pos_frame.append(spk_frame_right_map)
        self._spk_pos_frame.append(spk_frame_left_map)

        circular_data_pts = np.linspace(0, 2*np.pi, 30)
        for tf_idx in range(self._n_trigger_zones):
            zone_center_x = self._trigger_zones[tf_idx][0]
            zone_center_y = self._trigger_zones[tf_idx][1]
            zone_center_r = self._trigger_zones[tf_idx][2]

            # Generate a set of points at a distance r from the center
            zone_boundary_x = zone_center_x + zone_center_r * np.cos(circular_data_pts)
            zone_boundary_y = zone_center_y + zone_center_r * np.sin(circular_data_pts)

            # TODO: Try changing this to not animated and see if it still works out.
            trg_frame, = self._spk_pos_ax.plot(zone_boundary_x, zone_boundary_y, \
                    color='black', animated=True)
            self._spk_pos_frame.append(trg_frame)

        # Keeping the last two entries velocity and position frame so that it
        # is easy to manage them in animation.
        self._spk_pos_frame.append(pos_frame)
        self._spk_pos_frame.append(vel_frame)

        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_position_and_spike_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Spike-Position frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_place_field_fig(self):
        """
        Initialize figure window for dynamically showing place fields.
        """
        if self._pf_ax is None:
            return

        # For an open field, get build a heat map for place field.
        # For a linear track, show the field for the two different directions.
        if self._is_linear_track:
            right_map, = self._pf_ax.plot(self._pf_xmarkers, np.zeros_like(self._pf_xmarkers), \
                    color='C0', lw=3.0, animated=True)
            left_map, = self._pf_ax.plot(self._pf_xmarkers, np.zeros_like(self._pf_xmarkers), \
                    color='C3', lw=3.0, animated=True)
            self._pf_frame.append(right_map)
            self._pf_frame.append(left_map)
        else:
            pf_heatmap = self._pf_ax.imshow(np.zeros((PositionAnalysis.N_POSITION_BINS[0], \
                    PositionAnalysis.N_POSITION_BINS[1]), dtype='float'), vmin=0, \
                    vmax=self.__MAX_FIRING_RATE, aspect='auto', animated=True)
            self._pf_frame.append(pf_heatmap)

        # self.figure.colorbar(pf_heatmap)
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_place_field_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Place field frame created!')
        self._anim_objs.append(anim_obj)

    def run(self):
        """
        Start a GUI, launch all the graphics windows that have been requested
        in separate threads.
        """

        """
        self._keep_running.set()

        for p__thread in self._thread_list:
            p__thread.start()
        """

        # Start all the animation sources.
        self.playAnimation()

        # This is a blocking command... After you exit this, everything will end.
        while self._keep_running.is_set():
            time.sleep(1.0)

        # Stop all the animation sources.
        self.pauseAnimation()

        # Join all the fetcher threads.
        for p__thread in self._thread_list:
            p__thread.join()

        logging.info(MODULE_IDENTIFIER + "Terminated GUI and display pipes")
