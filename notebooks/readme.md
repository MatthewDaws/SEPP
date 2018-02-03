# Notebooks

Visualisations of the algorithms, using Chicago data.


- `0 - The input data` Look at the data we'll use.


### `Grid Sepp`

Grid based models where the trigger has no spatial component.

- A grid is laid down over the study area, and each event is assigned to the grid cell containing it.
- The model assumes there is no interaction between different grid cells.
- The background rate is assumed constant in each grid cell, but can vary between cells.
- The triggering process is assumed to be the same for each grid cell

We implement four models for the trigger:

- Simple exponential decay
- Exponential decay with an "inhibition" close to 0.
- A "histogram estimator" (fitted using a full derivation of the EM algorithm, and with an approximation, which is often more numerically stable).
- A kernel density estimation method motivated by the histogram estimator.


### `Grid and Space Sepp`

Background is still estimated from a grid, but the trigger takes account of possible interactions between all events (that is, the grid plays no part in the trigger process).

We implement trigger models of different complexity:

- Exponential decay in time, and Gaussian in space (but constant on a disk of chosen radius).
- Exponential decay in time, and constant on a disk in space.
- Histogram estimator in time, and constant on a disk in space.
- KDE in time and space separately
- KDE in time and space jointly (that is, consider a 3 dimensional spacetime)

For some models, we also implement the "stochastic EM algorithm" and observe that it often performs rather poorly (that is, differently from the full EM algorithm) on real data.


### `Full`

The background rate is now estimated with a KDE method.  We implement models with the time and space components being independent, and models with time and space being linked.  Fixed bandwidth, and nearest-neighbour based variable bandwidth models are available.


### `Fixed`

Models where the trigger kernel _shapes_ are fixed (by the user) but the background rate and the overall trigger rate are fitted using the EM algorithm.


### `Hawkes processes`

 A look at predicting Hawkes processes.


### `predictions`

Uses the `open_cp.scripted` library to make predictions (in python scripts, designed to be run overnight, say) and analyse to result (these predictions methods, on our data, are really no better than classical methods).
