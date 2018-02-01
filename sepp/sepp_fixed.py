"""
sepp_fixed.py
~~~~~~~~~~~~~

Fixed triggering kernel, so optimise just the rate and the background, in
various ways.
"""

import numpy as _np
from . import sepp_grid_space
import open_cp.sepp_base as sepp_base
import open_cp.kernels as kernels

class TimeKernel():
    """Abstract base class of a time kernel."""
    def __call__(self, times):
        """The time kernel, which should be a one dimensional probability
        kernel.

        :param times: One dimensional array of positive numbers.

        :return: One dimensional array of same size as input.
        """
        raise NotImplementedError()


class ExpTimeKernel(TimeKernel):
    """Exponentially decaying in time,
        :math:`f(x) = \omega e^{-\omega x}`

    :param omega: The "rate" of exponential.
    """
    def __init__(self, omega):
        self._omega = omega

    def __call__(self, times):
        return self._omega * _np.exp(-self._omega * _np.asarray(times))

    def __repr__(self):
        return "ExpTimeKernel(omega={})".format(self._omega)


class SpaceKernel():
    """Abstract base class of a space kernel."""
    def __call__(self, points):
        """The space kernel, which should be a two dimensional probability
        kernel.

        :param points: Two dimensional array of positive numbers, of shape
          `(2,N)`

        :return: One dimensional array of length `N`.
        """
        raise NotImplementedError()


class GaussianSpaceKernel(SpaceKernel):
    """Two-dimensional Gaussian decay.
        :math:`f(x) = (2\pi\sigma^2)^{-1} \exp(-\|x\|^2/2\sigma^2)`

    :param sigma: Standard deviation.
    """
    def __init__(self, sigma):
        self._sigmasq = 2 * sigma**2
        self._s = sigma

    def __call__(self, points):
        points = _np.asarray(points)
        dd = points[0]**2 + points[1]**2
        return _np.exp(-dd / self._sigmasq) / (_np.pi * self._sigmasq)

    def __repr__(self):
        return "GaussianSpaceKernel(sigma={})".format(self._s)



#############################################################################
# Grid based background estimation
#############################################################################

class GridModel(sepp_grid_space.Model, sepp_base.FastModel):
    """Grid based background estimation, with variable triggering rate."""
    def __init__(self, mu, T, grid, theta, time_kernel, space_kernel):
        super().__init__(mu, T, grid)
        self._theta = theta
        self._f = time_kernel
        self._g = space_kernel
        
    @property
    def theta(self):
        """The overall trigger rate."""
        return self._theta

    @property
    def time_kernel(self):
        return self._f

    @property
    def space_kernel(self):
        return self._g

    def time_trigger(self, times):
        return self._theta * self._f(times)
        
    def space_trigger(self, space_points):
        return self._g(space_points)

    def trigger(self, trigger_point, delta_points):
        delta_points = sepp_grid_space._atleast2d(delta_points)
        return self._theta * self._f(delta_points[0]) * self._g(delta_points[1:,:])

    def __repr__(self):
        return "GridModel(mu size={}, T={}, theta={}, f={}, g={})".format(
                self.mu.shape, self.T, self._theta, self._f, self._g)


class GridOptimiser(sepp_grid_space.Optimiser):
    def __init__(self, model, points):
        super().__init__(model, points)

    def theta_opt(self):
        return self.p_upper_tri_sum / self.num_points

    def iterate(self):
        return GridModel(mu=self.mu_opt(self.model.grid), T=self.model.T,
                      grid=self.model.grid, theta=self.theta_opt(),
                      time_kernel=self.model.time_kernel,
                      space_kernel=self.model.space_kernel)


class GridTrainer(sepp_grid_space.Trainer):
    def __init__(self, grid, time_kernel, space_kernel):
        super().__init__(grid)
        self._f = time_kernel
        self._g = space_kernel

    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        return GridModel(mu=mu, T=T, grid=self.grid, theta=0.5,
            time_kernel=self._f, space_kernel=self._g)

    @property
    def _optimiser(self):
        return GridOptimiser


class GridOptimiserFixedTheta(GridOptimiser):
    def theta_opt(self):
        return self.model.theta


class GridTrainerFixedTheta(GridTrainer):
    def __init__(self, grid, time_kernel, space_kernel, theta):
        super().__init__(grid, time_kernel, space_kernel)
        self._theta = theta

    def make_fixed(self, times):
        return _np.max(times)
    
    def initial_model(self, T, data):
        mu = self._initial_mu(data, T) / self.cell_area
        return GridModel(mu=mu, T=T, grid=self.grid, theta=self._theta,
            time_kernel=self._f, space_kernel=self._g)

    @property
    def _optimiser(self):
        return GridOptimiserFixedTheta
    


#############################################################################
# Combined space/time kernels
#############################################################################

# TODO??



#############################################################################
# KDE background estimation
#############################################################################

class KDEModel(sepp_base.ModelBase, sepp_base.FastModel):
    """KDE for the background"""
    def __init__(self, T, mu, background_kernel, theta, time_kernel, space_kernel):
        self._T = T
        self._mu = mu
        self._background_kernel = background_kernel
        self._theta = theta
        self._f = time_kernel
        self._g = space_kernel

    @property
    def T(self):
        """Total length of time.  Our convention is that timestamps will be
        in the interval `[0,T]`."""
        return self._T

    @property
    def mu(self):
        """Overall background rate."""
        return self._mu

    @property
    def background_kernel(self):
        return self._background_kernel

    @property
    def theta(self):
        """Overall trigger rate."""
        return self._theta

    @property
    def time_kernel(self):
        return self._f

    @property
    def space_kernel(self):
        return self._g

    def time_trigger(self, times):
        return self._theta * self._f(times)
        
    def space_trigger(self, space_points):
        return self._g(space_points)

    def trigger(self, trigger_point, delta_points):
        delta_points = sepp_grid_space._atleast2d(delta_points)
        return self._theta * self._f(delta_points[0]) * self._g(delta_points[1:,:])

    def background(self, points):
        space_points = sepp_grid_space._atleast2d(points)[1:,:]
        return self._background_kernel(space_points)

    def background_in_space(self, points):
        return self._background_kernel(points)

    def __repr__(self):
        return "KDEModel(T={}, mu={}, background={}, theta={}, f={}, g={}".format(
            self.T, self.mu, self.background_kernel, self.theta, self._f, self._g)


class KDEOptimiser(sepp_base.Optimiser):
    """Optimiser.  No edge correction.  Base class which is augmented by
    factory classes.
    
    :param model: Instance of :class:`Model`
    :param points: Array of shape `(3,N)`
    """
    def __init__(self, model, points):
        super().__init__(model, points)

    def mu_opt(self):
        return _np.sum(self.p_diag) / self.model.T

    def theta_opt(self):
        return self.p_upper_tri_sum / self.num_points

    def background_opt(self):
        raise NotImplementedError()

    def iterate(self):
        return KDEModel(T=self.model.T, mu=self.mu_opt(), theta=self.theta_opt(),
            background_kernel=self.background_opt(),
            time_kernel=self.model.time_kernel, space_kernel=self.model.space_kernel)


class KDEOptimiserFactory():
    def __init__(self, background_provider):
        self._background_provider = background_provider

    def __call__(self, model, points):
        opt = self._Optimiser(model, points)
        opt.background_provider = self._background_provider
        #p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        #opt.inject_pmatrix(p)
        return opt

    class _Optimiser(KDEOptimiser):
        def __init__(self, *args):
            super().__init__(*args)

        def background_opt(self):
            w = self.p_diag
            x = self.points[1:,:]
            return self.background_provider(x, w)


class KDETrainer(sepp_base.Trainer):
    """Training class
    
    :param background_provider: Instance of :class:`sepp_full.KernelProvider`
    """
    def __init__(self, time_kernel, space_kernel, background_provider):#, p_cutoff=99.9):
        super().__init__()
        self._f = time_kernel
        self._g = space_kernel
        self._opt_factory = KDEOptimiserFactory(background_provider)
        #self._opt_factory.pcutoff = p_cutoff

    def make_fixed(self, times):
        return _np.max(times)

    def initial_model(self, T, data):
        bk = kernels.GaussianBase(data[1:,:])
        return KDEModel(T=T, mu=data.shape[-1] / T, theta=0.5,
            background_kernel=bk, time_kernel=self._f, space_kernel=self._g)

    @property
    def _optimiser(self):
        return self._opt_factory


class KDEOptimiserFactoryFixedTheta(KDEOptimiserFactory):
    class _Optimiser(KDEOptimiser):
        def __init__(self, *args):
            super().__init__(*args)

        def theta_opt(self):
            return self.model.theta

        def background_opt(self):
            w = self.p_diag
            x = self.points[1:,:]
            return self.background_provider(x, w)


class KDETrainerFixedTheta(KDETrainer):
    def __init__(self, time_kernel, space_kernel, background_provider, theta):#, p_cutoff=99.9):
        super().__init__(time_kernel, space_kernel, background_provider)
        self._opt_factory = KDEOptimiserFactoryFixedTheta(background_provider)
        self._theta = theta
        #self._opt_factory.pcutoff = p_cutoff

    def initial_model(self, T, data):
        bk = kernels.GaussianBase(data[1:,:])
        return KDEModel(T=T, mu=data.shape[-1] / T, theta=self._theta,
            background_kernel=bk, time_kernel=self._f, space_kernel=self._g)
