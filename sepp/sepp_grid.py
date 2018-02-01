"""
sepp_grid
~~~~~~~~~

Grid based SEPP methods
"""

import numpy as _np
import datetime
import open_cp.predictors
import open_cp.data
import logging

_logger = logging.getLogger(__name__)

def _cell_iter(shape):
    return zip(*[coords.flat for coords in _np.indices(shape)])


class ConcreteBoundedGrid(open_cp.data.BoundedGrid):
    """Subclass of :class:`open_cp.data.BoundedGrid` which has an
    `extent` (and not just virtual methods)."""
    def __init__(self, xsize, ysize, xoffset, yoffset, xextent, yextent):
        super().__init__(xsize, ysize, xoffset, yoffset)
        self._extent = (xextent, yextent)

    @property
    def xextent(self):        
        return self._extent[0]

    @property
    def yextent(self):        
        return self._extent[1]


class SEPPGridTrainer(open_cp.predictors.DataTrainer):
    """Base class for grid based SEPP methods.

    :param grid: Take settings from this grid.
    :param timeunit: The time-unit to use to convert timestamps to numbers.
      Defaults to one day.
    """
    def __init__(self, grid, timeunit = datetime.timedelta(days=1)):
        self.time_unit = timeunit
        self.grid = grid

    @property
    def time_unit(self):
        """The time unit with which we convert the timestamps with."""
        return self._timeunit

    @time_unit.setter
    def time_unit(self, v):
        self._timeunit = _np.timedelta64(v)

    @property
    def grid(self):
        """The :class:`BoundedGrid` defining the cells."""
        return self._grid

    @grid.setter
    def grid(self, v):
        self._grid = ConcreteBoundedGrid(xsize = v.xsize, ysize = v.ysize,
            xoffset = v.xoffset, yoffset = v.yoffset,
            xextent = v.xextent, yextent = v.yextent)

    def to_cells(self, cutoff):
        """Convert the held data into a collection of time series, one for
        each cell.  We use the convention that times are negative, with `T=0`.
        So our :math:`t_i` is what in a formal writeup we'd refer to as
        :math:`-(T - t_i) = t_i - T`.
        
        :param cutoff: Use only data with a timestamp (strictly) before this
          time, and use this as the end time.

        :return: Array of shape `(yextent, xextent)`, the same size as the
          :attr:`grid`, 
        """
        cells = _np.empty((self.grid.yextent, self.grid.xextent), dtype=_np.object)
        for x in range(self.grid.xextent):
            for y in range(self.grid.yextent):
                cells[y, x] = list()
                    
        cutoff = _np.datetime64(cutoff)
        points = self.data[self.data.timestamps < cutoff]
        times = (points.timestamps - cutoff) / self.time_unit
        for t, x, y in zip(times, self.data.xcoords, self.data.ycoords):
            gx, gy = self.grid.grid_coord(x, y)
            cells[gy, gx].append(t)
        for cell in _cell_iter(cells.shape):
            cells[cell] = _np.asarray(cells[cell])
        return cells

    def prediction_from_background(self, model):
        """Using the grid stored in this class, and `model`, construct a
        prediction using only the background rate.
        
        :param model: Instance of :class:`ModelBase` to get the background
          rate from.
        """
        return open_cp.predictors.GridPredictionArray(
            xsize=self.grid.xsize, ysize=self.grid.ysize,
            xoffset=self.grid.xoffset, yoffset=self.grid.yoffset,
            matrix = _np.array(model.mu))

    def prediction(self, model, points, time_start=0, time_end=1, samples=100):
        """Using the grid stored in this class, and `model`, construct a
        prediction using the background rate and trigger.  We do this
        by sampling at 100 places in the time interval and returning the
        mean intensity.
        
        :param model: Instance of :class:`ModelBase` to get the background
          rate from.
        :param points: Array of same shape as `mu`, giving times of events
          in each grid cell.
        :param time_start: Using the convention that events occurred in the
          past, relative to `T=0`, give the start time for the prediction.
          Defaults to 0.
        :param time_end: The end time for the prediction, defaults to 1.
        """
        if not time_start < time_end:
            raise ValueError("Need `time_start` < `time_end`")
        matrix = _np.array(model.mu)
        t = _np.linspace(time_start, time_end, samples)
        for cell in _cell_iter(model.mu.shape):
            time_diffs = t[:,None] - points[cell][None,:]
            trig = model.trigger(cell, time_diffs.flatten()).reshape(time_diffs.shape)
            matrix[cell] += _np.mean(_np.sum(trig, axis=1))
        return open_cp.predictors.GridPredictionArray(
            xsize=self.grid.xsize, ysize=self.grid.ysize,
            xoffset=self.grid.xoffset, yoffset=self.grid.yoffset,
            matrix = matrix)

    def make_points(self, cutoff=None):
        """Helper method to generate the collection of points, and a suitable
        `T` value.
        
        :param cutoff: If `None` use all the data with the final timestamp
          as the end of time.  Otherwise use this as the end of time, and limit
          data to being before this time.
        
        :return: `(points, T)`
        """
        if cutoff is None:
            cutoff = self.data.time_range[1]
        cutoff = _np.datetime64(cutoff)
        points = self.to_cells(cutoff)
        T = (cutoff - self.data.time_range[0]) / self.time_unit
        return points, T


class ModelBase():
    """Absract base class for models.

    :param mu: Matrix of background rates in each cell.
    :param T: Total length of time.  Our convention is that timestamps will be
      in the interval `[-T,0]`.
    """
    def __init__(self, mu, T):
        self._mu = _np.asarray(mu)
        self._T = T

    @property
    def mu(self):
        """Matrix of background rates in each cell."""
        return self._mu

    @property
    def T(self):
        """Total length of time.  Our convention is that timestamps will be
        in the interval `[-T,0]`."""
        return self._T

    def trigger(self, cell, tdelta):
        """Return the triggering intensity in the `cell`, for the `tdelta`
        time gap."""
        raise NotImplementedError()

    def with_mu(self, new_mu):
        """Return a new instance with a new value for `mu`."""
        raise NotImplementedError()


class OptimiserBase():
    """Base class for optimisation procedure (i.e. the EM algorithm).

    :param model: Instance of :class:`ModelBase`
    :param points: Array of lists/arrays of times, for example, as returned
      by the `to_cells` method.
    """
    def __init__(self, model, points):
        self._model = model
        points = _np.asarray(points)
        if points.shape != model.mu.shape:
            raise ValueError("Points and mu should be the same shape")
        fp = [_np.asarray(times) for times in points.flatten()]
        for x in fp:
            if not _np.all(x[1:] - x[:-1] >= 0):
                raise ValueError("Each collection of times should be increasing.")
        if not any(len(x)>0 for x in fp):
            raise ValueError("`points` is empty!")
        self._points = _np.asarray(fp).reshape(points.shape)
        self._pcache = {}

    @property
    def points(self):
        return self._points

    @property
    def model(self):
        return self._model

    @property
    def mu(self):
        return self._model.mu

    @property
    def T(self):
        return self._model.T

    @staticmethod
    def _normalise_pmatrix(p):
        return p / _np.sum(p, axis=0)[None,:]

    def pmatrix(self, cell):
        """Return the p matrix (upper triangular) for the given cell.

        :param cell: Should be an index object to :attr:`points`."""
        if cell in self._pcache:
            return self._pcache[cell]
        times = self._points[cell]
        size = len(times)
        p = _np.zeros((size, size), dtype=_np.float)
        for i, t in enumerate(times):
            if i > 0:
                p[:i,i] = self._model.trigger(cell, t - times[:i])
        m = self.mu[cell]
        for i in range(size):
            p[i,i] = m
        p = self._normalise_pmatrix(p)
        self._pcache[cell] = p
        return p

    def cell_iter(self):
        """Yield `cell` tuples which iterate over :attr:`mu`"""
        return _cell_iter(self.mu.shape)

    def mu_opt(self):
        """Return the EM optimised estimate for mu"""
        mu = _np.empty_like(self.mu, dtype=_np.float)
        for cell in self.cell_iter():
            mu[cell] = _np.sum(_np.diag(self.pmatrix(cell))) / self.T
        return mu

    def p_upper_tri_sum(self, cell):
        p = self.pmatrix(cell)
        return sum (_np.sum(p[:i,i]) for i in range(1, p.shape[0]))


class OptimiserBaseWithRepeats(OptimiserBase):
    """Subclass of :class:`OptimiserBase` where we allow "repeated events",
    that is, :math:`t_i = t_j` for :math:`i \not= j`.  We do this by setting
    :math:`p_{i,j} = 0` for such pairs, which is consistent with the EM
    algorithm."""

    def pmatrix(self, cell):
        """Return the p matrix (upper triangular) for the given cell.

        :param cell: Should be an index object to :attr:`points`."""
        if cell in self._pcache:
            return self._pcache[cell]
        times = self._points[cell]
        size = len(times)
        p = _np.zeros((size, size), dtype=_np.float)
        for i, t in enumerate(times):
            if i > 0:
                p[:i,i] = self._model.trigger(cell, t - times[:i])
                mask = ~((t - times[:i]) <= 0)
                p[:i,i] *= mask
        m = self.mu[cell]
        for i in range(size):
            p[i,i] = m
        p = self._normalise_pmatrix(p)
        self._pcache[cell] = p
        return p
    



#############################################################################
# Exp model stuff
#############################################################################

class ExpDecayTrainer(SEPPGridTrainer):
    """Train the grid based sepp model, where the triggering kernel has the
    form
      :math:`g(t) = \theta \omega e^{-\omega t}`

    :param grid: The grid to assign points to.
    """
    def __init__(self, grid, allow_repeats=False, **kwargs):
        super().__init__(grid, **kwargs)
        self._allow_repeats = allow_repeats

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
        return points, ExpDecayModel(mu, T, 0.5, omega)

    def train(self, cutoff=None, iterations=10, use_fast=True):
        """Train the model.

        :param cutoff: If `None` use all the data with the final timestamp
          as the end of time.  Otherwise use this as the end of time, and limit
          data to being before this time.
        :param use_fast: If `True` then use the "fast" algorithm (no edge
          correction).
        """
        points, model = self.initial_model(cutoff)
        for _ in range(iterations):
            if use_fast:
                opt = ExpDecayOptFast(model, points, self._allow_repeats)
            else:
                opt = ExpDecayOpt(model, points, self._allow_repeats)
            model = opt.optimised_model()
            _logger.debug("Current model: %s", model)
        return model


class ExpDecayOptFast():
    """Optimise without edge correction."""
    def __init__(self, model, points, allow_repeats=False):
        if allow_repeats:
            self._base = OptimiserBaseWithRepeats(model, points)
        else:    
            self._base = OptimiserBase(model, points)

    def pmatrix(self, cell):
        return self._base.pmatrix(cell)

    def mu_opt(self):
        return self._base.mu_opt()

    @property
    def total_event_count(self):
        return sum (len(self._base.points[cell]) for cell in self._base.cell_iter())

    def theta_opt(self):
        return sum(self._base.p_upper_tri_sum(cell) for cell in self._base.cell_iter()) / self.total_event_count
            
    def omega_opt(self):
        over = sum(self._base.p_upper_tri_sum(cell) for cell in self._base.cell_iter())
        under = 0
        for cell in self._base.cell_iter():
            p = self._base.pmatrix(cell)
            pts = self._base.points[cell]
            for j in range(1, p.shape[0]):
                for i in range(j):
                    under += p[i,j] * (pts[j] - pts[i])
        return over / under

    def optimised_model(self):
        return ExpDecayModel(self._base.mu_opt(), self._base.T, self.theta_opt(), self.omega_opt())


class ExpDecayOpt():
    """Optimise with edge correction."""
    def __init__(self, model, points, allow_repeats=False):
        if allow_repeats:
            self._base = OptimiserBaseWithRepeats(model, points)
        else:    
            self._base = OptimiserBase(model, points)

    @property
    def points(self):
        return self._base.points

    def pmatrix(self, cell):
        return self._base.pmatrix(cell)

    def mu_opt(self):
        return self._base.mu_opt()

    def theta_opt(self):
        under = sum( _np.sum(1 - _np.exp(self._base.points[cell] * self._base.model.omega))
            for cell in self._base.cell_iter() )
        return sum(self._base.p_upper_tri_sum(cell) for cell in self._base.cell_iter()) / under
            
    def omega_opt(self):
        over = sum(self._base.p_upper_tri_sum(cell) for cell in self._base.cell_iter())
        under = 0
        for cell in self._base.cell_iter():
            p = self.pmatrix(cell)
            pts = self._base.points[cell]
            for j in range(1, p.shape[0]):
                for i in range(j):
                    under += p[i,j] * (pts[j] - pts[i])
            under += self._base.model.theta * _np.sum(-_np.exp(pts * self._base.model.omega) * pts)
        return over / under

    def optimised_model(self):
        return ExpDecayModel(self.mu_opt(), self._base.T, self.theta_opt(), self.omega_opt())


class ExpDecayModel(ModelBase):
    """Triggering kernel is of the form
      :math:`g(t) = \theta \omega e^{-\omega t}`
    """
    def __init__(self, mu, T, theta, omega):
        super().__init__(mu, T)
        self._theta = theta
        self._omega = omega

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    def with_mu(self, new_mu):
        """Return a new instance with a new value for `mu`."""
        return ExpDecayModel(new_mu, self.T, self.theta, self.omega)

    def with_theta(self, new_theta):
        """Return a new instance with a new value for `theta`."""
        return ExpDecayModel(self.mu, self.T, new_theta, self.omega)

    def with_omega(self, new_omega):
        """Return a new instance with a new value for `omega`."""
        return ExpDecayModel(self.mu, self.T, self.theta, new_omega)

    def trigger(self, cell, tdelta):
        return _np.exp(- self._omega * tdelta) * self._omega * self._theta

    def log_likelihood(self, points):
        """Compute the log likelihood of the data, given the parameters in this
        model.

        :param points: Array the same shape as :attr:`mu` each entry of which
          is a list of times in `[-T,0]`
        """
        points = _np.asarray(points).flatten()
        ll = 0.0
        for rate, times in zip(self.mu.flatten(), points):
            times = _np.asarray(times)
            for i, t in enumerate(times):
                if i == 0:
                    ll += _np.log(rate)
                else:
                    ll += _np.log(rate + _np.sum(self.trigger(None, t - times[:i])))
        ll -= _np.sum(self.mu.flatten()) * self.T
        for times in points:
            times = _np.asarray(times)
            ll -= self.theta * _np.sum(1 - _np.exp(self._omega * times))
        return ll

    def __repr__(self):
        return "ExpDecayModel(T={}, theta={}, omega={}, mu size={}".format(
            self.T, self.theta, self.omega, self.mu.shape)


##############################################################################
# Exponential decay model with a cutoff
#
# Could, I guess, have used subclassing, but didn't...
##############################################################################

class ExpDecayModelWithCutoff(ModelBase):
    """Triggering kernel is of the form
      :math:`g(t) = \theta \omega e^{-\omega t}`
    if :math:`t >= t_0` and `0` otherwise.  This inhibits triggering in a
    certain window of time after an event.

    :param cutoff: The value of :math:`t_0`.
    """
    def __init__(self, mu, T, theta, omega, cutoff):
        super().__init__(mu, T)
        self._theta = theta
        self._omega = omega
        self._t0 = cutoff

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    @property
    def cutoff(self):
        return self._t0

    def with_mu(self, new_mu):
        """Return a new instance with a new value for `mu`."""
        return ExpDecayModel(new_mu, self.T, self.theta, self.omega, self.cutoff)

    def with_theta(self, new_theta):
        """Return a new instance with a new value for `theta`."""
        return ExpDecayModel(self.mu, self.T, new_theta, self.omega, self.cutoff)

    def with_omega(self, new_omega):
        """Return a new instance with a new value for `omega`."""
        return ExpDecayModel(self.mu, self.T, self.theta, new_omega, self.cutoff)

    def trigger(self, cell, tdelta):
        tdelta = _np.asarray(tdelta)
        m = tdelta < self._t0
        out = _np.empty_like(tdelta, dtype=_np.float)
        out[m] = 0
        out[~m] = _np.exp(- self._omega * (tdelta[~m] - self._t0)) * self._omega * self._theta
        return out

    def trigger_integral(self, tdelta):
        tdelta = _np.asarray(tdelta)
        m = tdelta < self._t0
        out = _np.empty_like(tdelta, dtype=_np.float)
        out[m] = 0
        out[~m] = 1 - _np.exp(- self._omega * (tdelta[~m] - self._t0))
        return out

    def log_likelihood(self, points):
        """Compute the log likelihood of the data, given the parameters in this
        model.

        :param points: Array the same shape as :attr:`mu` each entry of which
          is a list of times in `[-T,0]`
        """
        points = _np.asarray(points).flatten()
        ll = 0.0
        for rate, times in zip(self.mu.flatten(), points):
            times = _np.asarray(times)
            for i, t in enumerate(times):
                if i == 0:
                    ll += _np.log(rate)
                else:
                    ll += _np.log(rate + _np.sum(self.trigger(None, t - times[:i])))
        ll -= _np.sum(self.mu.flatten()) * self.T
        for times in points:
            times = _np.asarray(times)
            ll -= _np.sum(self.trigger_integral(-times)) * self._theta
        return ll

    def __repr__(self):
        return "ExpDecayModel(T={}, theta={}, omega={}, mu size={}, t0={}".format(
            self.T, self.theta, self.omega, self.mu.shape, self._t0)


class ExpDecayOptFastWithCutoff(OptimiserBase):
    """Optimise without edge correction."""
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        under = 0
        for cell in self.cell_iter():
            under += _np.sum(- self.points[cell] >= self.model.cutoff)
        return sum(self.p_upper_tri_sum(cell) for cell in self.cell_iter()) / under
            
    def omega_opt(self):
        over = sum(self.p_upper_tri_sum(cell) for cell in self.cell_iter())
        under = 0
        for cell in self.cell_iter():
            p = self.pmatrix(cell)
            pts = self.points[cell]
            for j in range(1, p.shape[0]):
                for i in range(j):
                    under += p[i,j] * (pts[j] - pts[i] - self.model.cutoff)
        return over / under

    def optimised_model(self):
        return ExpDecayModelWithCutoff(self.mu_opt(), self.T, self.theta_opt(), self.omega_opt(), self.model.cutoff)


class ExpDecayOptWithCutoff(OptimiserBase):
    """Optimise with edge correction."""
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        under = 0
        for cell in self.cell_iter():
            T = -self.points[cell] - self.model.cutoff
            T[T<0] = 0
            under += _np.sum(1 - _np.exp(-T * self.model.omega))
        if under <= 0:
            for cell in self.cell_iter():
                T = -self.points[cell] - self.model.cutoff
                T[T<0] = 0
            raise AssertionError()
        return sum(self.p_upper_tri_sum(cell) for cell in self.cell_iter()) / under
            
    def omega_opt(self):
        over = sum(self.p_upper_tri_sum(cell) for cell in self.cell_iter())
        under = 0
        for cell in self.cell_iter():
            p = self.pmatrix(cell)
            pts = self.points[cell]
            for j in range(1, p.shape[0]):
                for i in range(j):
                    under += p[i,j] * (pts[j] - pts[i] - self.model.cutoff)
            T = -pts - self.model.cutoff
            T[T<0] = 0
            under += self.model.theta * _np.sum(_np.exp(-T * self.model.omega) * T)
        return over / under

    def optimised_model(self):
        return ExpDecayModelWithCutoff(self.mu_opt(), self.T, self.theta_opt(), self.omega_opt(), self.model.cutoff)


class ExpDecayTrainerWithCutoff(SEPPGridTrainer):
    """As :class:`ExpDecayTrainer` but with a cutoff on the triggering
    kernel.
    """
    def __init__(self, grid, cutoff, **kwargs):
        super().__init__(grid, **kwargs)
        self._t0 = cutoff

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
        return points, ExpDecayModelWithCutoff(mu, T, 0.5, omega, self._t0)

    def train(self, cutoff=None, iterations=10, use_fast=True):
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
                opt = ExpDecayOptFastWithCutoff(model, points)
            else:
                opt = ExpDecayOptWithCutoff(model, points)
            model = opt.optimised_model()
            _logger.debug("Current model: %s", model)
        return model
