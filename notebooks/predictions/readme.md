# Predictions

Using our models to my predictions, and scoring using hit-rate.

- [Produce Preds](produce_preds.py) : Python script to produce predictions; edit to change the "side"
- [Produce Preds Fixed](produce_preds_fixed.py) : Same, but just for the fixed trigger kernel models, with a large range of parameters.

- [Reload north](Reload%20North.ipynb) : Looks at the results for the North Side.
- [Reload north fixed](Reload%20north%20fixed.ipynb) : Look at just the results for the fixed trigger kernel models.
- [Reload south](Reload%20South.ipynb) : Looks at the results for the South Side.
- [Reload south fixed](Reload%20south%20fixed.ipynb) : Look at just the results for the fixed trigger kernel models.
- [Making predictions](Making predictions.ipynb) : An initial look at making predictions (from when we started to build the prediction code).
- [Making predictions 2](Making predictions 2.ipynb) : A more systematic look at prediction making.  In particular, we compare "fast predictions" (approximating kernels by histograms) with full predictions, and find there is little difference, except in terms of speed.

