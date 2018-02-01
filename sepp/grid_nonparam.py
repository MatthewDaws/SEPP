"""
grid_nonparam
~~~~~~~~~~~~~

Non-parametric, grid based SEPP method
"""

import numpy as _np
from . import sepp_grid
from . import histogram
import open_cp.kernels as kernels
import open_cp.sepp_base as sepp_base
import logging
_logger = logging.getLogger(__name__)

class NonParamModel(sepp_grid.ModelBase):
    """Non-parametric based triggering kernel
      :math:`f(x) = \sum_{r\geq0} \alpha_r \chi_{[hr,h(r+1))}(x)`
    where :math:`h` is the "bandwidth".
    """
    def __init__(self, mu, T, theta, bandwidth, alpha):
        super().__init__(mu, T)
        self._theta = theta
        self._h = bandwidth
        self._alpha = _np.asarray(alpha)
        if not len(self._alpha.shape) == 1:
            raise ValueError("alpha should be a one dimensional array.")
            
    @property
    def bandwidth(self):
        return self._h

    @property
    def theta(self):
        return self._theta

    @property
    def alpha(self):
        return self._alpha

    def trigger(self, cell, tdelta):
        tdelta = _np.asarray(tdelta)
        assert _np.all(tdelta >= 0)
        max_range = self._h * self._alpha.shape[0]
        m = tdelta < max_range
        out = _np.empty(tdelta.shape)
        indices = _np.floor_divide(tdelta[m], self._h).astype(_np.int)
        out[m] = self._alpha[indices]
        out[~m] = 0
        return out * self._theta / self._h

    def __repr__(self):
        if len(self.alpha) < 10:
            astr = str(self.alpha).replace("\n", "")
        else:
            astr = str(self.alpha[:10]).replace("\n", "")[:-1] + " ... ]"
        return "ExpDecayModel(T={}, theta={}, bandwidth={}, alpha={}".format(
            self.T, self.theta, self.bandwidth, astr)


class NonParamModelOpt(sepp_grid.OptimiserBaseWithRepeats):
    """Full optimisation"""
    def __init__(self, model, points):
        super().__init__(model, points)
        self._hist = None

    def alpha_opt(self):
        return self._get_hist().alpha

    def theta_opt(self):
        hist = self._get_hist()
        alpha = self.model.alpha
        gamma = hist.gamma
        le = min(alpha.shape[0], gamma.shape[0])
        under = _np.sum(alpha[:le] * gamma[:le])
        over = _np.sum(hist.beta)
        return over / under * self.model._h
    
    def _get_hist(self):
        if self._hist is not None:
            return self._hist
        x, p, t = [], [], []
        for cell in self.cell_iter():
            pmat = self.pmatrix(cell)
            pts = self.points[cell]
            for j in range(pmat.shape[1]):
                for i in range(j):
                    p.append(pmat[i,j])
                    x.append(pts[j] - pts[i])
            t.extend(pts)
        t = - _np.asarray(t)
        self._hist = histogram.EdgeCorrectHistogram(p, x, t, bandwidth=self.model._h, theta=self.model._theta)
        return self._hist
        
    def optimised_model(self):
        return NonParamModel(self.mu_opt(), self.T, self.theta_opt(), self.model.bandwidth, self.alpha_opt())


class NonParamModelOptFast(sepp_grid.OptimiserBase):
    """Without edge correction"""
    def __init__(self, model, points):
        super().__init__(model, points)
        self._hist = None

    def alpha_opt(self):
        return self._get_hist().alpha

    @property
    def total_event_count(self):
        return sum (len(self.points[cell]) for cell in self.cell_iter())

    def theta_opt(self):
        hist = self._get_hist()
        over = _np.sum(hist.beta)
        return over / self.total_event_count
    
    def _get_hist(self):
        if self._hist is not None:
            return self._hist
        x, p = [], []
        for cell in self.cell_iter():
            pmat = self.pmatrix(cell)
            pts = self.points[cell]
            for j in range(pmat.shape[1]):
                for i in range(j):
                    p.append(pmat[i,j])
                    x.append(pts[j] - pts[i])
        self._hist = histogram.NonEdgeCorrectHistogram(p, x, bandwidth=self.model._h)
        return self._hist
        
    def optimised_model(self):
        return NonParamModel(self.mu_opt(), self.T, self.theta_opt(), self.model.bandwidth, self.alpha_opt())


class NonParamTrainer(sepp_grid.SEPPGridTrainer):
    """Train a grid based model with histogram estimator for the
    triggering kernel.
    
    :param bandwidth: For the histogram estimator.
    """
    def __init__(self, grid, bandwidth, **kwargs):
        super().__init__(grid, **kwargs)
        self._h = bandwidth

    def initial_model(self, cutoff=None):
        """Return a suitable initial condition for the optimiser.

        :param cutoff: If `None` use all the data with the final timestamp
          as the end of time.  Otherwise use this as the end of time, and limit
          data to being before this time.

        :return: Pair `(points, model)` where `model` is an instance of
          :class:`ExpDecayModel`
        """
        points, T = self.make_points(cutoff)
        mu = _np.vectorize(len)(points) / T
        alen = T / self._h
        a = _np.exp(-_np.arange(alen) * self._h)
        a = a / _np.sum(a)
        return points, NonParamModel(mu, T, theta=0.5, bandwidth=self._h, alpha=a)

    def train(self, cutoff=None, iterations=10, use_fast=False):
        """Train the model.

        :param cutoff: If `None` use all the data with the final timestamp
          as the end of time.  Otherwise use this as the end of time, and limit
          data to being before this time.
        :param use_fast: If `True` then use the "fast" algorithm (no edge
          correction).
        """
        points, model = self.initial_model(cutoff)
        _logger.debug("Initial model: %s", model)
        for _ in range(iterations):
            if use_fast:
                opt = NonParamModelOptFast(model, points)
            else:
                opt = NonParamModelOpt(model, points)
            model = opt.optimised_model()
            _logger.debug("Current model: %s", model)
        return model



#############################################################################
# KDE trigger
#############################################################################

class KDEModel(sepp_grid.ModelBase):
    """KDE based trigger.
    """
    def __init__(self, mu, T, theta, f):
        super().__init__(mu, T)
        self._theta = theta
        self._f = f
            
    @property
    def trigger_func(self):
        return self._f

    @property
    def theta(self):
        return self._theta

    def trigger(self, cell, tdelta):
        return self._f(tdelta) * self._theta

    def __repr__(self):
        return "ExpDecayModel(T={}, theta={}, f={}".format(
            self.T, self.theta, self._f)


class KDEOpt(sepp_grid.OptimiserBaseWithRepeats):
    """Fixed bandwidth KDE estimation."""
    def __init__(self, model, points, bandwidth):
        super().__init__(model, points)
        self._h = bandwidth

    def theta_opt(self):
        theta = 0
        for cell in self.cell_iter():
            p = self.pmatrix(cell)
            for j in range(1, p.shape[1]):
                theta += _np.sum(p[:j,j])
        return theta / sum(len(pts) for pts in self.points.flat)

    def _times_probs_for_func_opt(self):
        probs = []
        times = []
        for cell in self.cell_iter():
            p = self.pmatrix(cell)
            pts = self.points[cell]
            for j in range(1, p.shape[1]):
                for i in range(j):
                    probs.append(p[i,j])
                    times.append(pts[j] - pts[i])
        return _np.asarray(times), _np.asarray(probs)
        
    def func_opt(self):
        times, probs = self._times_probs_for_func_opt()
        
        ker = kernels.GaussianBase(times)
        ker.covariance_matrix = 1
        ker.bandwidth = self._h
        ker.weights = probs
        return kernels.Reflect1D(ker)

    def optimised_model(self):
        return KDEModel(self.mu_opt(), self.T, self.theta_opt(), self.func_opt())


class KDEOptKNN(KDEOpt):
    """Use variable bandwidth by computing the distance to the
    kth nearest neighbour."""
    def __init__(self, model, points, number_neighbours):
        super().__init__(model, points, None)
        self._knn = number_neighbours
        
    def func_opt(self):
        times, probs = self._times_probs_for_func_opt()
        ker = kernels.GaussianNearestNeighbour(times, self._knn)
        ker.weights = probs
        return kernels.Reflect1D(ker)


class KDEProvider():
    """Provide different KDE methods."""
    def make_opt(self, model, points):
        raise NotImplementedError()
        
        
class KDEProviderFixedBandwidth(KDEProvider):
    """Uses :class:`KDEOpt`"""
    def __init__(self, bandwidth):
        self._h = bandwidth
        
    @property
    def bandwidth(self):
        """The fixed bandwidth we're using for the KDE."""
        return self._h

    def make_opt(self, model, points):
        return KDEOpt(model, points, self._h)

    def __repr__(self):
        return "KDE(h={})".format(self._h)
    

class KDEProviderKthNearestNeighbour(KDEProvider):
    """Uses :class:`KDEOptKNN`"""
    def __init__(self, k):
        self._k = k
        
    @property
    def k(self):
        """The number of nearest neighbours which we're using."""
        return self._k

    def make_opt(self, model, points):
        return KDEOptKNN(model, points, self._k)

    def __repr__(self):
        return "KDEnn(k={})".format(self._k)


class KDETrainer(sepp_grid.SEPPGridTrainer):
    """Train a grid based model with a KDE for the trigger.
    
    :param provider: Instance of :class:`KDEProvider`.
    """
    def __init__(self, grid, provider, **kwargs):
        super().__init__(grid, **kwargs)
        self._provider = provider

    @property
    def provider(self):
        """The KDE provider in use."""
        return self._provider

    def initial_model(self, cutoff=None):
        """Return a suitable initial condition for the optimiser.

        :param cutoff: If `None` use all the data with the final timestamp
          as the end of time.  Otherwise use this as the end of time, and limit
          data to being before this time.

        :return: Pair `(points, model)` where `model` is an instance of
          :class:`ExpDecayModel`
        """
        points, T = self.make_points(cutoff)
        mu = _np.vectorize(len)(points) / T
        omega = self.time_unit / _np.timedelta64(1, "D")
        def initial_func(t):
            return omega * _np.exp(-t*omega)
        return points, KDEModel(mu, T, theta=0.5, f=initial_func)

    def train(self, cutoff=None, iterations=10):
        """Train the model.

        :param cutoff: If `None` use all the data with the final timestamp
          as the end of time.  Otherwise use this as the end of time, and limit
          data to being before this time.
        """
        points, model = self.initial_model(cutoff)
        _logger.debug("Initial model: %s", model)
        for _ in range(iterations):
            opt = self._provider.make_opt(model, points)
            model = opt.optimised_model()
            _logger.debug("Current model: %s", model)
        return model
