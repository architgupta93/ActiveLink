"""
Constant definitions that can be accessed across files without having to
respecify them at multiple locations
"""

# Certain parameters can be tuned by just selecting the mode here.
MODE_ADJUSTING = True

# Spike sampling rate
SPIKE_SAMPLING_FREQ = 30000.0
INV_SPIKE_SAMPLING_FREQ = 1.0/SPIKE_SAMPLING_FREQ

# LFP sampling rate, and filter parameters
LFP_FREQUENCY     = 1500.0
LFP_FILTER_LENGTH = 0.002     # 1ms long window for filtering ripples
LFP_FILTER_ORDER  = int(LFP_FILTER_LENGTH * LFP_FREQUENCY)
LFP_BUFFER_TIME   = 0.5
LFP_BUFFER_LENGTH = int(LFP_FREQUENCY * LFP_BUFFER_TIME) # 1.0s long display window

# Calibration plot
CALIB_PLOT_WINDOW_LENGTH = 0.01
CALIB_PLOT_BUFFER_TIME   = 0.5
CALIB_MEASURE_TIME       = 0.1
CALIB_MEASURE_LENGTH     = int(CALIB_MEASURE_TIME / CALIB_PLOT_WINDOW_LENGTH)
CALIB_PLOT_BUFFER_LENGTH = int(CALIB_PLOT_BUFFER_TIME / CALIB_PLOT_WINDOW_LENGTH)

# Spike density estimation
SDE_TIME_BIN_SIZE = 0.005
SDE_UPDATE_FREQUENCY = int(1.0/SDE_TIME_BIN_SIZE)
SDE_BUFFER_TIME = 0.5
SDE_BUFFER_LENGTH = int(SDE_BUFFER_TIME/SDE_TIME_BIN_SIZE)
SDE_REFRACTORY_PERIOD = 0.075
SDE_SMOOTHING_FACTOR = 0.8

# Frequency range for Ripple. Keep in mind that a high range completely gets
# rid of the sharp wave ripple - It is a little harder to see the filtered
# ripple events and tell if they were real ripples or not!
RIPPLE_LO_FREQ    = 150 
RIPPLE_HI_FREQ    = 250
SHARPWAVE_LO_FREQ = 10
SHARPWAVE_HI_FREQ = 50
SHARPWAVE_FILTER_ORDER = 3

# Refractory period for a ripple (in seconds)
RIPPLE_SMOOTHING_WINDOW  = int(0.01 * LFP_FREQUENCY) # 10ms smoothing window
RIPPLE_SMOOTHING_FACTOR  = 0.2
RIPPLE_REFRACTORY_PERIOD = 0.075
RIPPLE_POWER_BUFFER_LENGTH = int(LFP_BUFFER_LENGTH / LFP_FILTER_ORDER)
INTERRUPT_REFRACTORY_PERIOD = 1.0
STAT_ADJUSTMENT_PERIOD   = 300
SWR_STAT_ADJUSTMENT_DATA_PTS = int(STAT_ADJUSTMENT_PERIOD * LFP_FREQUENCY)
SDE_STAT_ADJUSTMENT_DATA_PTS = int(STAT_ADJUSTMENT_PERIOD * SDE_UPDATE_FREQUENCY)

# Custom constants
LFP_ANIMATION_INTERVAL = 5     # Animation Frame rate (this is the delay between frames)
RIPPLE_POWER_THRESHOLD = 5     # Multiplicative factor for calling something a ripple
SDE_ZSCORE_THRESHOLD   = 5
MOVE_VELOCITY_THRESOLD = 5.0

if MODE_ADJUSTING:
    RIPPLE_SMOOTHING_FACTOR = 0.8
    RIPPLE_POWER_THRESHOLD = 5
    SDE_SMOOTHING_FACTOR = 0.8
    SDE_ZSCORE_THRESHOLD = 5
else:
    RIPPLE_SMOOTHING_FACTOR = 0.2
    RIPPLE_POWER_THRESHOLD = 3
    SDE_SMOOTHING_FACTOR = 0.2
    SDE_ZSCORE_THRESHOLD = 3

# Plotting and aesthetics
N_POWER_BINS = 20
INTERRUPTION_WINDOW = 0.25    # Time before/after interruption displayed
INTERRUPTION_TPTS   = int(INTERRUPTION_WINDOW * LFP_FREQUENCY)

# Specify the size of the field here in cm
FIELD_SIZE = (200, 200)
