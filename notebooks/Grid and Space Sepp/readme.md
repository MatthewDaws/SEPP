# Grid and Space SEPP

Using a grid based background model, but with triggering not liked to the grid.


### General notebooks

- [Kernel musings](Kernel%20musings.ipynb) : A very quick think about different kernels; not explored further.


### Models fitted to simulated data

We simulate data from a known SEPP model, and see how the various fitting algorithms perform (they generally perform well).

- [Simulated data](Simulated%20data.ipynb) : For the base models
- [Simulated data space uniform](Simulated%20data%20space%20uniform.ipynb) : With a uniform trigger: events are uniformly triggered in a disk in space
- [Simulated data KDE](Simulated%20data%20KDE.ipynb) : Same, but with the trigger kernel estimated using a KDE method.


### With real data

Again from Chicago.

- [With real data](With%20real%20data.ipynb) : Uses `sepp_grid_space.Trainer1`
- [With real data simple model](With%20real%20data%20simple%20model.ipynb) : Uses `sepp_grid_space.Trainer2`
- [With real data histogram time](With%20real%20data%20histogram%20time.ipynb) : Uses `sepp_grid_space.Trainer3`
- [With real data KDE](With%20real%20data%20KDE.ipynb) : Uses `sepp_grid_space.Trainer4`
- [With real data KDE 1](With%20real%20data%20KDE%201.ipynb) : Uses `sepp_grid_space.Trainer5`

