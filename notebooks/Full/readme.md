# Full

Estimate both the background intensity and the trigger intensity using a variety of KDE methods.

- [`Simulated Data.ipynb`](Simulated%20Data.ipynb) : Fit to artificial data.
- [`Initial Conditions Matter.ipynb`](Initial%20Conditions%20Matter.ipynb) : Explores the problem that,
  initially, when I compared the "full" method with the "grid" method, I obtained radically different
  results.  This was eventually traced down (with help from the techniques displayed in this notebook)
  to differences in initial conditions.

## Using real data

- [`With real data.ipynb`](With%20real%20data.ipynb) : Uses a wide bandwidth for the background; explores the full EM algorithm and the stochastic EM algorithm.
- [`With real data 1.ipynb`](With%20real%20data%201.ipynb) : The same, but with a narrower bandwidth for the background KDE.
- [`With real data, split.ipynb`](With%20real%20data,%20split.ipynb) : Uses a trigger kernel KDE which is separate in space and time.
- [`With real data, split 1.ipynb`](With%20real%20data,%20split%201.ipynb) : Same, with different bandwidth settings.
- [`With real data, split 2.ipynb`](With%20real%20data,%20split%202.ipynb) : Same, with different bandwidth settings.

## With modified data

As for the real data, but on the dataset of "redistributed" points, using `opencrimedata`.
