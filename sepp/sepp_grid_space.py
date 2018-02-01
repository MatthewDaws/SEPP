"""
sepp_grid_space.py
~~~~~~~~~~~~~~~~~~

Background is estimated from a grid, but the trigger kernel is fully spatial.

We use the `open_cp` code, which unfortunately has a different convention
for the times: now they start at 0.
"""

import open_cp.sepp_base as sepp_base
from . import sepp_grid
from . import histogram
import open_cp.kernels as kernels
import numpy as _np
import logging
_logger = logging.getLogger(__name__)

def _atleast2d(points):
    points = _np.asarray(points)
    if len(points.shape) == 1:
        points = points[:,None]
    return points


class _GridMixin():
    @property
    def grid(self):
        """The :class:`BoundedGrid` defining the cells."""
        return self._grid

    @grid.setter
    def grid(self, v):
        self._grid = sepp_grid.ConcreteBoundedGrid(xsize = v.xsize, ysize = v.ysize,
            xoffset = v.xoffset, yoffset = v.yoffset,
            xextent = v.xextent, yextent = v.yextent)

    def points_to_grid(self, points):
        """Yield pairs (gx, gy) of grid coords for the points, in order.
        
        :param points: Array of `[times, x, y]`
        """
        for pt in points.T:
            yield self.grid.grid_coord(pt[1], pt[2])
            
    @property
    def cell_area(self):
        return self.grid.xsize * self.grid.ysize
        

class Model(sepp_base.ModelBase, _GridMixin):
    """Base model for models which estimate the background risk via a grid.
        
    :param mu: Matrix of background rates in each cell.
    :param T: Total length of time.  Our convention is that timestamps will be
      in the interval `[-T,0]`.
    :param grid: Grid to take settings from; for background rate.
    """
    def __init__(self, mu, T, grid):
        self._mu = _np.asarray(mu)
        self._T = T
        self.grid = grid

    @property
    def mu(self):
        """Matrix of background rates in each cell."""
        return self._mu

    @property
    def T(self):
        """Total length of time.  Our convention is that timestamps will be
        in the interval `[0,T]`."""
        return self._T

    def background(self, points):
        points = _atleast2d(points)
        return _np.asarray([
            self._mu[gy, gx] for gx, gy in self.points_to_grid(points)])

    def background_in_space(self, space_points):
        """Use by :class:`sepp_base.FastModel` but we can support out of the
        box as our background doesn't vary in time."""
        gx, gy = self.grid.grid_coord(*space_points)
        return self._mu[gy, gx]
    
    def __repr__(self):
        return "Model(mu size={}, T={})".format(self.mu.shape, self.T)


class Optimiser(sepp_base.Optimiser):
    """Base optimiser which handles the background rate.
    
    :param model: Instance of :class:`Model`
    :param points: Array of shape `(3,N)`
    """
    def __init__(self, model, points):
        super().__init__(model, points)

    def mu_opt(self, grid):
        """Optimise for mu.
        
        :param grid: Grid object to take settings from; used for assigning
          points to grid cells.
        """
        mu = _np.zeros((grid.yextent, grid.xextent))
        for j, pt in enumerate(self.points.T):
            gx, gy = grid.grid_coord(pt[1], pt[2])
            mu[gy, gx] += self.p[j,j]
        area = grid.xsize * grid.ysize
        return mu / (area * self.model.T)

    def theta_no_edge_opt(self):
        """Assuming no edge effects, return the optimisation for theta."""
        return self.p_upper_tri_sum / self.num_points

    def inject_pmatrix(self, p):
        """Manually set a pmatrix, for testing."""
        assert p.shape == self._p.shape
        self._p = p


class Trainer(sepp_base.Trainer, _GridMixin):
    """Base training class.
    
    :param grid: Grid to take settings from, for background rate.
    """
    def __init__(self, grid):
        super().__init__()
        self.grid = grid

    def _initial_mu(self, points, T):
        mu = _np.zeros((self.grid.yextent, self.grid.xextent))
        for gx, gy in self.points_to_grid(points):
            mu[gy,gx] += 1
        return mu / T
    
    def to_predictor(self, model):
        return sepp_base.Predictor(self.grid, model)

    def sample_to_points(self, model, predict_time):
        """Returned sampled `(background_points, trigger_deltas)`."""
        _, data = self.make_data(predict_time)        
        opt = self._optimiser(model, data)
        return opt.sample_to_points()



#############################################################################
# Exponential decay in time, "capped" gaussian in space
#############################################################################

class Model1(Model, sepp_base.FastModel):
    def __init__(self, mu, T, grid, theta, omega, alpha, sigmasq, r0):
        super().__init__(mu, T, grid)
        if omega <= 0 or sigmasq <= 0 or r0 <= 0 or r0**2 * alpha * _np.pi > 1:
            raise ValueError("Invalid parameter")
        self._theta = theta
        self._omega = omega
        self._alpha = alpha
        self._sigma = sigmasq
        self._r0 = r0
        
    @property
    def theta(self):
        return self._theta
    
    @property
    def omega(self):
        return self._omega
    
    @property
    def alpha(self):
        return self._alpha

    @property
    def sigma(self):
        return _np.sqrt(self._sigma)

    @property
    def sigmasq(self):
        return self._sigma

    @property
    def r0(self):
        return self._r0
    
    @property
    def beta(self):
        under = 1 / (2 * _np.pi * self._sigma)
        under *= _np.exp(self._r0**2 / (2 * self._sigma))
        return (1 - _np.pi * self._alpha * self._r0**2) * under
    
    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        times = pts[0]
        distsq = (pts[1]**2 + pts[2]**2)
        
        intensity = self._omega * _np.exp(-self._omega * times)

        mask = distsq <= self.r0**2
        sp_int = _np.empty(intensity.shape)
        sp_int[mask] = self.alpha
        sp_int[~mask] = self.beta * _np.exp(-distsq[~mask] / (2 * self._sigma))
        
        return self._theta * intensity * sp_int

    def time_trigger(self, times):
        return self._theta * self._omega * _np.exp(-self._omega * times)
        
    def space_trigger(self, space_points):
        distsq = _np.asarray((space_points[0]**2 + space_points[1]**2))
        mask = distsq <= self.r0**2
        sp_int = _np.empty(distsq.shape)
        sp_int[mask] = self.alpha
        sp_int[~mask] = self.beta * _np.exp(-distsq[~mask] / (2 * self._sigma))
        return sp_int

    def space_trigger_mass(self, radius):
        mass = _np.pi * min(radius, self.r0)**2 * self.alpha
        if radius > self.r0:
            mass += 2 * _np.pi * self.beta * self.sigmasq * (
                    _np.exp(-self.r0**2 / (2 * self.sigmasq))
                    - _np.exp(-radius**2 / (2 * self.sigmasq)) )
        return mass

    def __repr__(self):
        return "Model1(mu size={}, T={}, grid={}, theta={}, omega={}, alpha={}, sigma^2={}, r0={}".format(
                self.mu.shape, self.T, self.grid.region(),
                self.theta, self.omega,
                self.alpha, self.sigmasq, self.r0)


class Optimiser1(Optimiser):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        dt = self.model.T - self.points[0]
        lower = _np.sum(1 - _np.exp(-self.model.omega * dt))
        return self.p_upper_tri_sum / lower

    def omega_opt(self):
        lower = sum(_np.sum(self.diff_col_times(c) * self.upper_tri_col(c)) for c in range(1, self.num_points))
        dt = self.model.T - self.points[0]
        lower += self.model.theta * _np.sum(dt * _np.exp(-self.model.omega * dt))
        return self.p_upper_tri_sum / lower

    @property
    def _abc_values(self):
        if hasattr(self, "_abc_cache"):
            return self._abc_cache
        a, b, c = 0, 0, 0
        for col in range(1, self.num_points):
            p = self.upper_tri_col(col)
            dspace = self.diff_col_points(col)
            rs = dspace[0]**2 + dspace[1]**2
            mask = rs <= self.model.r0**2
            a += _np.sum(p[mask])
            mask = ~mask
            b += _np.sum(p[mask])
            c += _np.sum(rs[mask] * p[mask])
        self._abc_cache = (a,b,c)
        return self._abc_cache
    
    def alpha_opt(self):
        a, b, _ = self._abc_values
        return a / (a + b) / (_np.pi * self.model.r0**2)
    
    def sigmasq_opt(self):
        _, b, c = self._abc_values
        return (c - b * self.model.r0**2) / (b + b)

    def iterate(self):
        return Model1(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      omega=self.omega_opt(), alpha=self.alpha_opt(),
                      sigmasq = self.sigmasq_opt(), r0=self.model.r0)


class Trainer1(Trainer):
    def __init__(self, grid, r0):
        super().__init__(grid)
        self._r0 = r0
        
    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        alpha = 1 / (4 * _np.pi * self._r0**2)
        return Model1(mu=mu, T=T, grid=self.grid, theta=0.5, omega=1,
                      alpha=alpha, sigmasq=max(1, self._r0**2), r0=self._r0)

    @property
    def _optimiser(self):
        return Optimiser1


#############################################################################
# Exponential decay in time, circular window in space
#############################################################################

class Model2(Model):
    def __init__(self, mu, T, grid, theta, omega, r0):
        super().__init__(mu, T, grid)
        self._theta = theta
        self._omega = omega
        self._r0 = r0
        
    @property
    def theta(self):
        return self._theta
    
    @property
    def omega(self):
        return self._omega
    
    @property
    def r0(self):
        return self._r0

    @property
    def alpha(self):
        return 1 / (_np.pi * self.r0**2)
    
    def time_trigger(self, times):
        return self._theta * self._omega * _np.exp(-self._omega * times)
        
    def space_trigger(self, space_points):
        distsq = _np.asarray((space_points[0]**2 + space_points[1]**2))
        mask = distsq <= self.r0**2
        sp_int = _np.zeros(distsq.shape)
        sp_int[mask] = self.alpha
        return sp_int

    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        times = pts[0]
        distsq = (pts[1]**2 + pts[2]**2)
        
        intensity = self.omega * _np.exp(-self.omega * times)

        mask = distsq <= self.r0**2
        sp_int = _np.zeros(intensity.shape)
        sp_int[mask] = self.alpha
        
        return self.theta * intensity * sp_int

    def space_trigger_mass(self, radius):
        return _np.pi * min(radius, self.r0)**2 * self.alpha

    def __repr__(self):
        return "Model2(mu size={}, T={}, grid={}, theta={}, omega={}, r0={}".format(
                self.mu.shape, self.T, self.grid.region(),
                self.theta, self.omega, self.r0)


class Optimiser2(Optimiser):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        dt = self.model.T - self.points[0]
        lower = _np.sum(1 - _np.exp(-self.model.omega * dt))
        return self.p_upper_tri_sum / lower

    def omega_opt(self):
        lower = sum(_np.sum(self.diff_col_times(c) * self.upper_tri_col(c)) for c in range(1, self.num_points))
        dt = self.model.T - self.points[0]
        lower += self.model.theta * _np.sum(dt * _np.exp(-self.model.omega * dt))
        return self.p_upper_tri_sum / lower

    def iterate(self):
        return Model2(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      omega=self.omega_opt(), r0=self.model.r0)


class Trainer2(Trainer):
    def __init__(self, grid, r0):
        super().__init__(grid)
        self._r0 = r0
        
    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        return Model2(mu=mu, T=T, grid=self.grid, theta=0.5, omega=1, r0=self._r0)

    @property
    def _optimiser(self):
        return Optimiser2


#############################################################################
# Histogram in time, circular window in space
#############################################################################

class Model3(Model):
    def __init__(self, mu, T, grid, theta, bandwidth, alpha, r0):
        super().__init__(mu, T, grid)
        self._theta = theta
        self._h = bandwidth
        self._r0 = r0
        self._alpha = _np.asarray(alpha)
        
    @property
    def theta(self):
        return self._theta
    
    @property
    def bandwidth(self):
        return self._h
    
    @property
    def r0(self):
        return self._r0
    
    @property
    def alpha_array(self):
        return self._alpha

    @property
    def alpha(self):
        return 1 / (_np.pi * self.r0**2)
    
    def time_trigger(self, times):
        indices = _np.floor_divide(times, self._h).astype(_np.int)
        mask = indices < len(self._alpha)
        intensity = _np.empty(len(times))
        intensity[mask] = self._alpha[indices[mask]] / self._h
        intensity[~mask] = 0
        return self._theta * intensity
        
    def space_trigger(self, space_points):
        distsq = _np.asarray((space_points[0]**2 + space_points[1]**2))
        mask = distsq <= self.r0**2
        sp_int = _np.zeros(distsq.shape)
        sp_int[mask] = self.alpha
        return sp_int
    
    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        times = pts[0]
        distsq = (pts[1]**2 + pts[2]**2)
        
        indices = _np.floor_divide(times, self._h).astype(_np.int)
        mask = indices < len(self._alpha)
        intensity = _np.empty(len(times))
        intensity[mask] = self._alpha[indices[mask]] / self._h
        intensity[~mask] = 0

        mask = distsq <= self.r0**2
        sp_int = _np.zeros(intensity.shape)
        sp_int[mask] = self.alpha
        
        return self.theta * intensity * sp_int

    def space_trigger_mass(self, radius):
        return _np.pi * min(radius, self.r0)**2 * self.alpha

    def __repr__(self):
        return "Model3(mu size={}, T={}, grid={}, theta={}, bandwidth={}, r0={}".format(
                self.mu.shape, self.T, self.grid.region(),
                self.theta, self._h, self.r0)


class Optimiser3(Optimiser):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        hist = self._get_hist()
        le = min(len(hist.alpha), len(hist.gamma))
        under = _np.sum(hist.alpha[:le] * hist.gamma[:le])
        return self.model.bandwidth * _np.sum(hist.beta) / under

    def alpha_opt(self):
        hist = self._get_hist()
        return hist.alpha        
        
    def _get_hist(self):
        if hasattr(self, "_hist"):
            return self._hist

        p, x = [], []
        for col in range(1, self.num_points):
            p.extend(self.upper_tri_col(col))
            x.extend(self.diff_col_times(col))
        t = self.model.T - self.points[0]

        self._hist = self._make_hist(p, x, t, self.model.bandwidth, self.model.theta)
        return self._hist
        
    def _make_hist(self, p, x, t, h, theta):
        return histogram.EdgeCorrectHistogram(p, x, t, h, theta)
    
    def iterate(self):
        return Model3(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      bandwidth=self.model.bandwidth, r0=self.model.r0,
                      alpha=self.alpha_opt())


class Optimiser3fast(Optimiser3):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        return self.theta_no_edge_opt()

    def _make_hist(self, p, x, t, h, theta):
        return histogram.NonEdgeCorrectHistogram(p, x, h)


class Trainer3(Trainer):
    def __init__(self, grid, r0, bandwidth, use_fast=False):
        super().__init__(grid)
        self._r0 = r0
        self._h = bandwidth
        if use_fast:
            self._opt = Optimiser3fast
        else:
            self._opt = Optimiser3
        
    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        a = _np.exp(-_np.arange(100))
        a = a / _np.sum(a)
        return Model3(mu=mu, T=T, grid=self.grid, theta=0.5, r0=self._r0,
                      bandwidth=self._h, alpha=a)

    @property
    def _optimiser(self):
        return self._opt


#############################################################################
# KDE in space and time, separately
#############################################################################

class Model4(Model, sepp_base.FastModel):
    def __init__(self, mu, T, grid, theta, time_kernel, space_kernel):
        super().__init__(mu, T, grid)
        self._theta = theta
        self._f = time_kernel
        self._g = space_kernel
        
    @property
    def theta(self):
        return self._theta
    
    @property
    def time_kernel(self):
        return self._f
    
    @property
    def space_kernel(self):
        return self._g
    
    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        times = pts[0]
        dists = pts[1:,:]
        return self.theta * self._f(times) * self._g(dists)

    def time_trigger(self, times):
        return self.theta * self._f(times)

    def space_trigger(self, space_points):
        return self._g(space_points)

    def __repr__(self):
        return "Model4(mu size={}, T={}, grid={}, theta={}, time={}, space={}".format(
                self.mu.shape, self.T, self.grid.region(),
                self.theta, self._f, self._g)


class Optimiser4(Optimiser):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        return self.p_upper_tri_sum / self.num_points
        
    def time_opt(self):
        p, x = [], []
        for col in range(1, self.num_points):
            p.extend(self.upper_tri_col(col))
            x.extend(self.diff_col_times(col))
        ker = self.time_kernel_provider(x, p)
        return kernels.Reflect1D(ker)

    def space_opt(self):
        p, x = [], []
        for col in range(1, self.num_points):
            p.extend(self.upper_tri_col(col))
            for pt in self.diff_col_points(col).T:
                x.append(pt)
        return self.space_kernel_provider(_np.asarray(x).T, p)
        
    def iterate(self):
        return Model4(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      time_kernel=self.time_opt(), space_kernel=self.space_opt())


class _Optimiser4Factory():
    def __init__(self, time_kernel_provider, space_kernel_provider, p_cutoff,
        optimiser_class=Optimiser4):
        self._ff = time_kernel_provider
        self._gg = space_kernel_provider
        self._p_cutoff = p_cutoff
        self._opt_class = optimiser_class
        
    def __call__(self, model, points):
        opt = self._opt_class(model, points)
        opt.time_kernel_provider = self._ff
        opt.space_kernel_provider = self._gg
        p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        opt.inject_pmatrix(p)
        return opt


class Trainer4(Trainer):
    def __init__(self, grid, time_kernel_provider, space_kernel_provider, p_cutoff=99.9,
                initial_time_scale=1, initial_space_scale=20, optimiser_class=Optimiser4):
        super().__init__(grid)
        self._fac = _Optimiser4Factory(time_kernel_provider, space_kernel_provider,
                p_cutoff, optimiser_class)
        self._initial_time_scale = initial_time_scale
        self._initial_space_scale = initial_space_scale
        
    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        def tk(t):
            return _np.exp(-self._initial_time_scale * _np.asarray(t))
        def sk(x):
            x = _np.asarray(x)
            r = _np.sqrt(x[0]**2 + x[1]**2)
            bw = 1 / self._initial_space_scale
            return bw * bw * _np.exp(-bw * r) / (2 * _np.pi)
        return Model4(mu=mu, T=T, grid=self.grid, theta=0.5,
                      time_kernel=tk, space_kernel=sk)

    @property
    def _optimiser(self):
        return self._fac


class Optimiser4a(Optimiser):
    """As :class:`Optimiser4` but using the stochastic EM algorithm."""
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        return self.p_upper_tri_sum / self.num_points
        
    def _sample_triggers(self):
        if hasattr(self, "_triggers"):
            return self._triggers
        _, self._triggers = self.sample_to_points()
        return self._triggers

    def time_opt(self):
        times = self._sample_triggers()[0]
        ker = self.time_kernel_provider(times, None)
        return kernels.Reflect1D(ker)

    def space_opt(self):
        pts = self._sample_triggers()[1:,:]
        return self.space_kernel_provider(pts, None)
        
    def iterate(self):
        return Model4(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      time_kernel=self.time_opt(), space_kernel=self.space_opt())


class Trainer4a(Trainer4):
    """As :class:`Trainer4` but using the stochastic EM algorithm.
    """
    def __init__(self, grid, time_kernel_provider, space_kernel_provider, p_cutoff=99.9,
                initial_time_scale=1, initial_space_scale=20):
        super().__init__(grid, time_kernel_provider, space_kernel_provider, p_cutoff,
            initial_time_scale, initial_space_scale, Optimiser4a)



#############################################################################
# KDE in space and time together
#############################################################################

class Model5(Model):
    def __init__(self, mu, T, grid, theta, trigger_kernel):
        super().__init__(mu, T, grid)
        self._theta = theta
        self._ker = trigger_kernel
        
    @property
    def theta(self):
        return self._theta
    
    @property
    def trigger_kernel(self):
        return self._ker

    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        return self.theta * self._ker(pts)

    def __repr__(self):
        return "Model5(mu size={}, T={}, grid={}, theta={}, trigger={}".format(
                self.mu.shape, self.T, self.grid.region(),
                self.theta, self._ker)

class Optimiser5(Optimiser):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        return self.p_upper_tri_sum / self.num_points
        
    def data_for_ker_opt(self):
        """For testing, return the `(x, p)` we use to form the KDE."""
        n = (self.num_points-1) * self.num_points // 2
        p, x = _np.empty(n), _np.empty((3,n))
        index = 0
        for col in range(1, self.num_points):
            p[index:index+col] = self.upper_tri_col(col)
            dif = self._points[:, col][:,None] - self._points[:, :col]
            x[:,index:index+col] = dif
            index += col
        return x, p
    
    def ker_opt(self):
        ker = self.kernel_provider(*self.data_for_ker_opt())
        return kernels.ReflectedKernel(ker, 0)
        
    def iterate(self):
        return Model5(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      trigger_kernel=self.ker_opt())


class _Optimiser5Factory():
    def __init__(self, kernel_provider, p_cutoff, optimiser_class=Optimiser5):
        self._ker_provider = kernel_provider
        self._p_cutoff = p_cutoff
        self._opt_class = optimiser_class
        
    def __call__(self, model, points):
        opt = self._opt_class(model, points)
        opt.kernel_provider = self._ker_provider
        p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        opt.inject_pmatrix(p)
        return opt


class Trainer5(Trainer):
    def __init__(self, grid, kernel_provider, p_cutoff=99.9,
                initial_time_scale=1, initial_space_scale=20, optimiser_class=Optimiser5):
        super().__init__(grid)
        self._fac = _Optimiser5Factory(kernel_provider, p_cutoff, optimiser_class)
        self._initial_time_scale = initial_time_scale
        self._initial_space_scale = initial_space_scale
        
    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        def tk(t):
            return _np.exp(-self._initial_time_scale * t) * self._initial_time_scale
        def sk(x):
            x = _np.asarray(x)
            rr = x[0]**2 + x[1]**2
            bwsq = self._initial_space_scale * self._initial_space_scale
            return _np.exp(-rr / (2 * bwsq)) / (2 * _np.pi * bwsq)
        def ker(pts):
            pts = _np.asarray(pts)
            return tk(pts[0]) * sk(pts[1:])
        return Model5(mu=mu, T=T, grid=self.grid, theta=0.5, trigger_kernel=ker)

    @property
    def _optimiser(self):
        return self._fac
