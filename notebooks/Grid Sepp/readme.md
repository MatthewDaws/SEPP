# Grid SEPP

Models which assume independent triggering in each grid cell.

### General notebooks

- [Example of grided model](Example%20of%20grided%20model.ipynb) : Run the model on artificial data and see how the quality of MLE fitting varies.
- [Explore timeseries of data](Explore%20timeseries%20of%20data.ipynb) : Look at how the events are distributed in time.
- [Likelihood](Likelihood.ipynb) : A very quick look at how the likelihood varies with parameters.
- [Making predictions](Making%20predictions.ipynb) : A very initial look at making predictions.
- [Making predictions 2](Making%20predictions%202.ipynb) : What has grown into a test-bed for all prediction methods; but plots and findings for the articles are in a different directory.
- [Southwest failure](Southwest%20failure.ipynb) : Exploring a failure of one of the optimisers, which is ultimately caused by exactly repeated timestamps (see the article for a full discussion).


### Models with original data

We consistently use data from Chicago, North Side.

- [With real data](With%20real%20data.ipynb) : Fit the model `sepp_grid.ExpDecayTrainer`
- [With real data inhibited exponential](With%20real%20data%20inhibited%20exponential.ipynb) : Fit the model `sepp_grid.ExpDecayTrainerWithCutoff`
- [With real data nonparametric](With%20real%20data%20nonparametric.ipynb) : Fit the model `grid_nonparam.NonParamTrainer`
- [With real data KDE](With%20real%20data%20KDE.ipynb) : Fit the model `grid_nonparam.KDETrainer`


### Models with modified data

Same again, but with adjusted data from `opencrimedata`.
