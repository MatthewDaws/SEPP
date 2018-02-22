# Fixed

SEPP algorithms where we fix the triggering kernel, and just estimate the background rate and overall trigger rate using the EM algorithm.

- [`Simulated data`](Simulated%20data.ipynb) : Fit the model to artificial data; as usual, it works well.


### With real data

We use expoential decay in time, and Gaussian decay in space.

- [`With real data, Grid background`](With%20real%20data,%20Grid%20background.ipynb) : Estimate the background rate using a grid.
- [`With real data, KDE background`](With%20real%20data,%20KDE%20background.ipynb) : Estimate the background rate using a (fixed bandwidth) KDE method.


### With modified data

Same again, using the data restributed using opencrimedata.

- [`With modified real data, Grid background`](With%20modified%20real%20data,%20Grid%20background.ipynb) : Estimate the background rate using a grid.
- [`With modified real data, KDE background`](With%20modified%20real%20data,%20KDE%20background.ipynb) : Estimate the background rate using a (fixed bandwidth) KDE method.
