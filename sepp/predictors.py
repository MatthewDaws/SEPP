"""
predictors.py
~~~~~~~~~~~~~

Make predictions using our SEPP implementations, and the `open_cp.scripted`
module.
"""

import open_cp.evaluation
import open_cp.sepp_base
from . import sepp_grid
from . import grid_nonparam
from . import sepp_grid_space
from . import sepp_fixed
from . import sepp_full
import logging as _logging
_logger = _logging.getLogger(__name__)
import open_cp.logger
open_cp.logger.log_to_stdout("sepp")

#################################################
# sepp_grid
#################################################

class ExpDecayGridProvider():
    """Use :class:`sepp_grid.ExpDecayTrainer`.  This generally performs badly,
    but is a useful benchmark.  A factory class which "trains" itself.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, cutoff=None, iterations=50):
        trainer = sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
        trainer.data = points
        self._model = trainer.train(cutoff, iterations)

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.model = self._model
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time):
            predictor = sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
            predictor.data = points
            cells, T = predictor.make_points(time)
            pred = predictor.prediction(self.model, cells)
            return pred
    
        def __repr__(self):
            return "ExpDecayGridProvider"
        
        @property
        def args(self):
            return ""


#################################################
# grid_nonparam
#################################################

class NonParamGridProvider():
    """Uses :class:`grid_nonparam.NonParamTrainer` with a histogram estimator.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param bandwidth: The width of each bar in the histogram estimator, in
      units of a day.
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, bandwidth, cutoff=None, iterations=50):
        trainer = grid_nonparam.NonParamTrainer(grid, bandwidth=bandwidth)
        trainer.data = points
        self._model = trainer.train(cutoff, iterations, use_fast=True)
        self._bw = bandwidth

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.model = self._model
        provider.bandwidth = self._bw
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time):
            predictor = grid_nonparam.NonParamTrainer(grid, bandwidth=self.bandwidth)
            predictor.data = points
            cells, T = predictor.make_points(time)
            pred = predictor.prediction(self.model, cells)
            return pred
    
        def __repr__(self):
            return "NonParamGridProvider(h={})".format(self.bandwidth)
        
        @property
        def args(self):
            return "{}".format(self.bandwidth)
    

class KDEGridProvider():
    """Uses :class:`grid_nonparam.KDETrainer` with a histogram estimator.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param kde_provider: The KDE provider, from :mod:`grid_nonparam`.
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, kde_provider, cutoff=None, iterations=50):
        self._kde = kde_provider
        trainer = grid_nonparam.KDETrainer(grid, self._kde)
        trainer.data = points
        _logger.debug("Training grid_nonparam.KDETrainer with %s", self._kde)
        self._model = trainer.train(cutoff, iterations)

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.model = self._model
        provider.kde = self._kde
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time):
            predictor = grid_nonparam.KDETrainer(grid, self.kde)
            predictor.data = points
            cells, T = predictor.make_points(time)
            pred = predictor.prediction(self.model, cells)
            return pred
    
        def __repr__(self):
            return "KDEGridProvider({})".format(self.kde)
        
        @property
        def args(self):
            return "{}".format(self.kde)


#################################################
# sepp_grid_space
#################################################

class GridSpaceExpDecayProvider():
    """Uses :class:`sepp_grid_space.Trainer1` which is exponential decay in
    time, and "capped" Gaussian in space.  Is a "type 1" predictor, so you
    should use `add_prediction_range` on the :class:`scripted.Data` instance.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param r0: The "cap" of the Gaussian decay.
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, r0, cutoff=None, iterations=50):
        self._r0 = r0
        trainer = sepp_grid_space.Trainer1(grid, self._r0)
        trainer.data = points
        _logger.debug("Training sepp_grid_space.Trainer1 with r0=%s", r0)
        model = trainer.train(cutoff, iterations)
        self._predictor = trainer.to_predictor(model).to_fast_split_predictor()

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.predictor = self._predictor
        provider.r0 = self._r0
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            # `grid` is already set
            predictor = self.predictor
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            return "GridSpaceExpDecayProvider(r0={})".format(self.r0)
        
        @property
        def args(self):
            return "{}".format(self.r0)


class GridSpaceSimpleProvider():
    """Uses :class:`sepp_grid_space.Trainer2` which is exponential decay in
    time, and uniform in a disc in space.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param r0: Radius of the spatial disc.
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, r0, cutoff=None, iterations=50):
        self._r0 = r0
        trainer = sepp_grid_space.Trainer2(grid, self._r0)
        trainer.data = points
        _logger.debug("Training sepp_grid_space.Trainer2 with r0=%s", r0)
        model = trainer.train(cutoff, iterations)
        self._predictor = trainer.to_predictor(model).to_fast_split_predictor()

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.predictor = self._predictor
        provider.r0 = self._r0
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            # `grid` is already set
            predictor = self.predictor
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            return "GridSpaceSimpleProvider(r0={})".format(self.r0)
        
        @property
        def args(self):
            return "{}".format(self.r0)


class GridSpaceSimpleHistogramProvider():
    """Uses :class:`sepp_grid_space.Trainer3` which is a histogram estimator
    in time, and uniform in a disc in space.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param r0: Radius of the spatial disc.
    :param bandwidth: The bandwidth for the histogram estimator.
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, r0, bandwidth, cutoff=None, iterations=50):
        self._r0 = r0
        self._h = bandwidth
        trainer = sepp_grid_space.Trainer3(grid, r0=self._r0, bandwidth=self._h, use_fast=True)
        trainer.data = points
        _logger.debug("Training sepp_grid_space.Trainer3 with r0=%s, bandwidth=%s", r0, bandwidth)
        model = trainer.train(cutoff, iterations)
        self._predictor = trainer.to_predictor(model).to_fast_split_predictor()

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.predictor = self._predictor
        provider.r0 = self._r0
        provider.h = self._h
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            # `grid` is already set
            predictor = self.predictor
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            return "GridSpaceSimpleHistogramProvider(r0={}, h={})".format(self.r0, self.h)
        
        @property
        def args(self):
            return "r0={}, h={}".format(self.r0, self.h)


class GridSpaceKDEProvider():
    """Uses :class:`sepp_grid_space.Trainer4` which is a KDE in time and in
    time, separately.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param time_kernel_provider: Instance of :class:`kernels.KernelProvider`
    :param space_kernel_provider: Instance of :class:`kernels.KernelProvider`
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, time_kernel_provider, space_kernel_provider,
            cutoff=None, iterations=50):
        self._tkp = time_kernel_provider
        self._skp = space_kernel_provider
        trainer = sepp_grid_space.Trainer4(grid, self._tkp, self._skp, p_cutoff=99.99)
        trainer.data = points
        _logger.debug("Training sepp_grid_space.Trainer4 with %s / %s", self._tkp, self._skp)
        model = trainer.train(cutoff, iterations)
        self._predictor = trainer.to_predictor(model).to_fast_split_predictor_histogram(time_bin_size=0.1, space_bin_size=10)

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.predictor = self._predictor
        provider.tkp = self._tkp
        provider.skp = self._skp
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            # `grid` is already set
            predictor = self.predictor
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            return "GridSpaceKDEProvider(tker={}, sker={})".format(self.tkp, self.skp)
        
        @property
        def args(self):
            return "tker={}, sker={}".format(self.tkp, self.skp)


### Missing is Trainer5, which doesn't "factor" in space/time, and so will be
### incredibly slow...

#############################################################################
# sepp_full
#############################################################################

class FullKDEProvider():
    """Uses :class:`sepp_full.Trainer1` which is a KDE in time and in
    time, separately; and a KDE for the background.

    :param grid: The grid to use for training (should be the same as that used
      for prediction: is actually only used in the approximation step to make
      forming predictions faster).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param time_kernel_provider: Instance of :class:`kernels.KernelProvider`
    :param space_kernel_provider: Instance of :class:`kernels.KernelProvider`
    :param background_kernel_provider: Instance of
      :class:`kernels.KernelProvider`
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, time_kernel_provider, space_kernel_provider,
            background_kernel_provider, cutoff=None, iterations=20):
        self._tkp = time_kernel_provider
        self._skp = space_kernel_provider
        self._bkp = background_kernel_provider
        opt_fac = sepp_full.Optimiser1Factory(background_kernel_provider,
                time_kernel_provider, space_kernel_provider)
        trainer = sepp_full.Trainer1(opt_fac)
        trainer.data = points
        _logger.debug("Training sepp_full.Trainer1 with %s / %s / %s", self._tkp, self._skp, self._bkp)
        model = trainer.train(cutoff, iterations)
        _logger.debug("Forming approximate histogram based predictor")
        predictor = open_cp.sepp_base.Predictor(grid, model)
        self._predictor = predictor.to_fast_split_predictor_histogram(time_bin_size=0.1, space_bin_size=10)

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.predictor = self._predictor
        provider.tkp = self._tkp
        provider.skp = self._skp
        provider.bkp = self._bkp
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            # `grid` is already set
            predictor = self.predictor
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            return "FullKDEProvider(tker={}, sker={}, bker={})".format(self.tkp, self.skp, self.bkp)
        
        @property
        def args(self):
            return "tker={}, sker={}, bker={}".format(self.tkp, self.skp, self.bkp)


#################################################
# sepp_fixed
#################################################

class GridFixedTriggerProvider():
    """Uses :class:`sepp_fixed.GridTrainer` which fixes the triggers and
    estimates the background and (optionally) the trigger rate.

    :param grid: The grid to use for training (should be the same as that used
      for predictions).
    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param time_kernel: Instance of :class:`sepp_fixed.TimeKernel`
    :param space_kernel: Instance of :class:`sepp_fixed.SpaceKernel`
    :param theta: If not `None` then fixed the trigger rate at this value
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, grid, points, time_kernel, space_kernel,
            theta=None, cutoff=None, iterations=50):
        self._tk = time_kernel
        self._sk = space_kernel
        self._theta = theta
        if theta is None:
            trainer = sepp_fixed.GridTrainer(grid, self._tk, self._sk)
            _logger.debug("Training sepp_fixed.GridTrainer with %s / %s", self._tk, self._sk)
        else:
            trainer = sepp_fixed.GridTrainerFixedTheta(grid, self._tk, self._sk, theta)
            _logger.debug("Training sepp_fixed.GridTrainerFixedTheta with %s / %s / %s", self._tk, self._sk, theta)
        trainer.data = points
        model = trainer.train(cutoff, iterations)
        self._predictor = trainer.to_predictor(model).to_fast_split_predictor()

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.predictor = self._predictor
        provider.tk = self._tk
        provider.sk = self._sk
        provider.theta = self._theta
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            # `grid` is already set
            predictor = self.predictor
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            if self.theta is None:
                return "GridFixedTriggerProvider(tker={}, sker={})".format(self.tk, self.sk)
            return "GridFixedTriggerProvider(tker={}, sker={}, theta={})".format(self.tk, self.sk, self.theta)
        
        @property
        def args(self):
            if self.theta is None:
                return "tker={}, sker={}".format(self.tkp, self.skp)
            return "tker={}, sker={}, theta={}".format(self.tkp, self.skp, self.theta)


class KDEFixedTriggerProvider():
    """Uses :class:`sepp_fixed.KDETrainer` which fixes the triggers and
    estimates the background using a KDE and (optionally) the trigger rate.

    :param points: The _training_ points, :class:`open_cp.data.TimedPoints`
      instances.
    :param time_kernel: Instance of :class:`sepp_fixed.TimeKernel`
    :param space_kernel: Instance of :class:`sepp_fixed.SpaceKernel`
    :param background_kernel_provider: Instance of
      :class:`sepp_full.KernelProvider`
    :param theta: If not `None` then fixed the trigger rate at this value
    :param cutoff: End time for training.
    :param iterations: Number of iterations to perform.
    """
    def __init__(self, points, time_kernel, space_kernel, background_kernel_provider,
            theta=None, cutoff=None, iterations=30):
        self._tk = time_kernel
        self._sk = space_kernel
        self._bkp = background_kernel_provider
        self._theta = theta
        if theta is None:
            trainer = sepp_fixed.KDETrainer(self._tk, self._sk, self._bkp)
            _logger.debug("Training sepp_fixed.KDETrainer with %s / %s / %s", self._tk, self._sk, self._bkp)
        else:
            trainer = sepp_fixed.KDETrainerFixedTheta(self._tk, self._sk, self._bkp, theta)
            _logger.debug("Training sepp_fixed.KDETrainerFixedTheta with %s / %s / %s / %s",
                    self._tk, self._sk, self._bkp, theta)
        trainer.data = points
        self._model = trainer.train(cutoff, iterations)

    def __call__(self, *args):
        provider = self._Provider(*args)
        provider.tk = self._tk
        provider.sk = self._sk
        provider.bkp = self._bkp
        provider.theta = self._theta
        provider.model = self._model
        return provider

    class _Provider(open_cp.evaluation.StandardPredictionProvider):
        def give_prediction(self, grid, points, time, end_time):
            predictor = open_cp.sepp_base.Predictor(grid, self.model).to_fast_split_predictor()
            predictor.data = points
            pred = predictor.predict(time, end_time, time_samples=5, space_samples=-5)
            return pred
    
        def __repr__(self):
            if self.theta is None:
                return "KDEFixedTriggerProvider(tker={}, sker={}, back={})".format(self.tk, self.sk, self.bkp)
            return "KDEFixedTriggerProvider(tker={}, sker={}, back={}, theta={})".format(self.tk, self.sk, self.bkp, self.theta)
        
        @property
        def args(self):
            if self.theta is None:
                return "tker={}, sker={}, back={}".format(self.tk, self.sk, self.bkp)
            return "tker={}, sker={}, back={}, theta={}".format(self.tk, self.sk, self.bkp, self.theta)
