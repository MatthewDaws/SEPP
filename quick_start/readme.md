# Quick-Start Guide

## Installation

We recommend using [Anaconda](https://www.anaconda.com/download/) python.

- Follow the [instructions](https://github.com/QuantCrimAtLeeds/PredictCode/blob/master/quick_start/install.md) to install the `open_cp` library and supporting libraries.
- Now install the `sepp` module, most easily done via:

      pip install https://github.com/MatthewDaws/SEPP/zipball/master


## The input data

At a minimum, you need to provide a list of events, in the form of x, y coordinates and timestamps.  The coordinates should be in meters, or another projected unit, and not Longitude and Latitude.  The timestamps need to be Python `datetime` objects (or `numpy` `datetime64` objects) and not just text strings.

- See the notebook [the input data](input_data.ipynb) for an example.
- Compare with [the guide](https://github.com/QuantCrimAtLeeds/PredictCode/blob/master/quick_start/scripted_intro.md#the-input-data) for `open_cp`.

### Geometry

You will probably also need an outline of our study area; most likely in shapefile format, or perhaps GeoJSON, or similar.  

- See the notebook [the input data](input_data.ipynb) for an example.


## Model types


