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

We detail the models we consider in both the paper, and in the "readme" files in the [notebooks](../notebooks) directory and sub-directories.  To summarise:

- `sepp.sepp_grid` for grid based models where the trigger has no spatial component.
- `sepp.sepp_grid_space` for models where the background is still estimated from a grid, but the trigger takes account of possible interactions between all events (that is, the grid plays no part in the trigger process).
- `sepp.full` for models where the background rate is now estimated with a KDE method.  We implement models with the time and space components being independent, and models with time and space being linked.  Fixed bandwidth, and nearest-neighbour based variable bandwidth models are available.
- `sepp.fixed` for models where the trigger kernel _shapes_ are fixed (by the user) but the background rate and the overall trigger rate are fitted using the EM algorithm.


## Making predictions

This is relatively slow, and so best done (we think) in a Python script, and not a
notebook.  The model `sepp.predictors` works with the `open_cp.scripted` module.

- See the [`open_cp.scripted` quick start guide](https://github.com/QuantCrimAtLeeds/PredictCode/blob/master/quick_start/scripted_intro.md).
- See the scripts in [notebooks/predictions](../notebooks/predictions) for examples which make predictions using Chicago data.
