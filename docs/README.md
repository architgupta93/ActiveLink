# ActiveLink
This project is an attempt to provide a single interface for real-time analysis and closed-loop feedback for neural activity.

## Instructions for setting up ActiveLink
ActiveLink is written entirely in python. Its modular structure should make it easy for users to add in their own processing modules for their own use cases.

I recommend using some kind of virtual environment for installing ActiveLink (conda works great). Some external python libraries are necessary for running the program (this list will grow as more features are released):

1. PyQt5 (for the interface)
2. Numpy
3. Scipy
4. OpenCV

## User guide
As is, ActiveLink is only compatible with SpikeGadgets supplied software [Trodes](https://spikegadgets.com/trodes/).
From the commandline, run

    $ python -O ActiveLink.py

This should open the main window.
The "-O" option runs the program in the most efficient mode. Running it without that results in extensive logging which while useful for debugging, slows down the program considerably.

![ActiveLink](OpeningScreen.png)

# Logging Infrastructure
Without connecting the program to an underlying data-acquisition system, it can be used for logging and tracking electrode movements.

# Real-time signal processing
Select "File > Connect" to launch a connection to [Trodes](https://spikegadgets.com/trodes/), the recording software provided by SpikeGadgets.

![RealTimeProcessing](StreamOptions.png)

# Online detection and disruption of Sharp-Wave Ripples
![OnlineRipples](SWRDetection.png)

## Upcoming Features

# Online clustering of single-unit activity

# Real-time Bayesian inference of spatial position from place-cell activity

