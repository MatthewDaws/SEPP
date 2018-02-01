"""
sepp_full.py
~~~~~~~~~~~~

Estimate both background and trigger using KDEs.

We use the `open_cp` code, which unfortunately has a different convention
for the times: now they start at 0.
"""

import open_cp.sepp_base as sepp_base
import open_cp.kernels as kernels
import numpy as _np
import logging
_logger = logging.getLogger(__name__)

def _atleast2d(points):
    points = _np.asarray(points)
    if len(points.shape) == 1:
        points = points[:,None]
    return points


class Model(sepp_base.ModelBase):
    """KDEs for background and trigger.
        
    :param T: Total length of time.  Our convention is that timestamps will be
      in the interval `[-T,0]`.
    :param mu: Overall background rate.
    :param background_kernel: Two dimensional kernel for the background.
    :param theta: Overall trigger rate.
    :param trigger_kernel: Three dimensional kernel for the trigger.
    """
    def __init__(self, T, mu, background_kernel, theta, trigger_kernel):
        self._T = T
        self._mu = mu
        self._background_kernel = background_kernel
        self._theta = theta
        self._trigger_kernel = trigger_kernel

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
    def trigger_kernel(self):
        return self._trigger_kernel

    def background(self, points):
        points = _atleast2d(points)
        return self._mu * self._background_kernel(points[1:,:])

    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        return self._theta * self._trigger_kernel(pts)

    def __repr__(self):
        return "Model(T={}, mu={}, background={}, theta={}, trigger={}".format(
            self.T, self.mu, self.background_kernel, self.theta, self.trigger_kernel)


class Optimiser(sepp_base.Optimiser):
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

    def trigger_opt(self):
        raise NotImplementedError()

    def inject_pmatrix(self, p):
        """Manually set a pmatrix, for testing."""
        assert p.shape == self._p.shape
        self._p = p

    def data_for_trigger_opt(self):
        """Compute the data we need for the trigger optimisation step.
        
        :return: `(points, weights)`.
        """
        x, w = [], []
        for i in range(1, self.num_points):
            w.extend(self.upper_tri_col(i))
            pts = self._points[:, i][:,None] - self._points[:, :i]
            x.extend(pts.T)
        return _np.asarray(x).T, _np.asarray(w)

    def iterate(self):
        return Model(T=self.model.T, mu=self.mu_opt(), theta=self.theta_opt(),
            background_kernel=self.background_opt(),
            trigger_kernel=self.trigger_opt())


class OptimiserFactory():
    """Provides an optimiser with differing KDE methods.  The trigger kernel is
    always reflected about 0 in time.
    
    :param background_provider: Instance of :class:`KernelProvider` to use for
      estimating the background (2D) kernel.
    :param trigger_provider: Instance of :class:`KernelProvider` to use for
      estimating the trigger (3D) kernel.
    """
    def __init__(self, background_provider, trigger_provider):
        self._background_provider = background_provider
        self._trigger_provider = trigger_provider

    def __call__(self, model, points):
        opt = self._Optimiser(model, points)
        opt.background_provider = self._background_provider
        opt.trigger_provider = self._trigger_provider
        p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        opt.inject_pmatrix(p)
        return opt

    @property
    def pcutoff(self):
        return self._p_cutoff

    @pcutoff.setter
    def pcutoff(self, v):
        self._p_cutoff = v

    class _Optimiser(Optimiser):
        def __init__(self, *args):
            super().__init__(*args)

        def background_opt(self):
            w = self.p_diag
            x = self.points[1:,:]
            return self.background_provider(x, w)

        def trigger_opt(self):
            ker = self.trigger_provider(*self.data_for_trigger_opt())
            return kernels.ReflectedKernel(ker, 0)


class OptimiserSEMFactory(OptimiserFactory):
    """As :class:`Optimiser` but uses the stochastic EM algorithm.
    
    :param background_provider: Instance of :class:`KernelProvider` to use for
      estimating the background (2D) kernel.
    :param trigger_provider: Instance of :class:`KernelProvider` to use for
      estimating the trigger (3D) kernel.
    """
    def __init__(self, background_provider, trigger_provider):
        super().__init__(background_provider, trigger_provider)

    def __call__(self, model, points):
        opt = self._Optimiser(model, points)
        opt.background_provider = self._background_provider
        opt.trigger_provider = self._trigger_provider
        p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        opt.inject_pmatrix(p)
        return opt

    class _Optimiser(Optimiser):
        def __init__(self, *args):
            super().__init__(*args)

        def _sample_points(self):
            if not hasattr(self, "_sampled_points"):
                self._sampled_points = self.sample_to_points()
            return self._sampled_points

        def background_opt(self):
            backs, _ = self._sample_points()
            return self.background_provider(backs[1:,:], None)

        def trigger_opt(self):
            _, trigs = self._sample_points()
            ker = self.trigger_provider(trigs, None)
            return kernels.ReflectedKernel(ker, 0)


class Trainer(sepp_base.Trainer):
    """Training class
    
    :param optimiser_factory: Factory to use to build the optimiser.
    """
    def __init__(self, optimiser_factory, p_cutoff=99.9,
            initial_time_scale=1, initial_space_scale=20):
        super().__init__()
        self._opt_factory = optimiser_factory
        self._opt_factory.pcutoff = p_cutoff
        self._initial_time_scale = initial_time_scale
        self._initial_space_scale = initial_space_scale

    def to_predictor(self, model):
        raise NotImplementedError()

    def sample_to_points(self, model, predict_time):
        """Returned sampled `(background_points, trigger_deltas)`."""
        _, data = self.make_data(predict_time)        
        opt = self._optimiser(model, data)
        return opt.sample_to_points()

    def make_fixed(self, times):
        return _np.max(times)

    def initial_model(self, T, data):
        bk = kernels.GaussianBase(data[1:,:])
        def tk(pts):
            pts = _np.asarray(pts)
            p = _np.exp(-self._initial_time_scale * pts[0]) * self._initial_time_scale
            rr = pts[1]**2 + pts[2]**2
            bwsq = self._initial_space_scale * self._initial_space_scale
            return p * _np.exp(-rr / (2 * bwsq)) / (2 * _np.pi * bwsq)
        return Model(T=T, mu=data.shape[-1] / T, theta=0.5,
            background_kernel=bk, trigger_kernel=tk)

    @property
    def _optimiser(self):
        return self._opt_factory


#############################################################################
# Trigger kernel split in time/space
#############################################################################

class Model1(sepp_base.ModelBase, sepp_base.FastModel):
    """KDEs for background and trigger; trigger now split.
        
    :param T: Total length of time.  Our convention is that timestamps will be
      in the interval `[-T,0]`.
    :param mu: Overall background rate.
    :param background_kernel: Two dimensional kernel for the background.
    :param theta: Overall trigger rate.
    :param trigger_time_kernel: One dimensional kernel for the time trigger.
    :param trigger_space_kernel: Two dimensional kernel for the space trigger.
    """
    def __init__(self, T, mu, background_kernel, theta, trigger_time_kernel, trigger_space_kernel):
        self._T = T
        self._mu = mu
        self._background_kernel = background_kernel
        self._theta = theta
        self._tk_time = trigger_time_kernel
        self._tk_space = trigger_space_kernel

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
    def trigger_time_kernel(self):
        return self._tk_time

    @property
    def trigger_space_kernel(self):
        return self._tk_space

    def background(self, points):
        points = _atleast2d(points)
        return self._mu * self._background_kernel(points[1:,:])

    def trigger(self, trigger_point, delta_points):
        pts = _atleast2d(delta_points)
        return self._theta * self._tk_time(pts[0]) * self._tk_space(pts[1:])

    def time_trigger(self, times):
        return self._theta * self._tk_time(times)

    def space_trigger(self, space_points):
        return self._tk_space(space_points)

    def background_in_space(self, space_points):
        return self._mu * self._background_kernel(space_points)

    def __repr__(self):
        return "Model1(T={}, mu={}, background={}, theta={}, trigger={},{}".format(
            self.T, self.mu, self.background_kernel, self.theta, self._tk_time, self._tk_space)


class Optimiser1Factory():
    """Provides an optimiser suitable for :class:`Model1`.  The trigger time
    kernel will always be reflected in time.
    
    :param background_provider: Instance of :class:`KernelProvider` to use for
      estimating the background (2D) kernel.
    :param trigger_provider: Instance of :class:`KernelProvider` to use for
      estimating the trigger (3D) kernel.
    """
    def __init__(self, background_provider, trigger_time_provider, trigger_space_provider):
        self._background_provider = background_provider
        self._trigger_time_provider = trigger_time_provider
        self._trigger_space_provider = trigger_space_provider
        
    def __call__(self, model, points):
        opt = self._Optimiser(model, points)
        opt.background_provider = self._background_provider
        opt.trigger_time_provider = self._trigger_time_provider
        opt.trigger_space_provider = self._trigger_space_provider
        p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        opt.inject_pmatrix(p)
        return opt

    @property
    def pcutoff(self):
        return self._p_cutoff

    @pcutoff.setter
    def pcutoff(self, v):
        self._p_cutoff = v

    class _Optimiser(Optimiser):
        def __init__(self, *args):
            super().__init__(*args)

        def background_opt(self):
            w = self.p_diag
            x = self.points[1:,:]
            return self.background_provider(x, w)

        def _cached_trigger_data(self):
            if not hasattr(self, "_tdcache"):
                self._tdcache = self.data_for_trigger_opt()
            return self._tdcache

        def trigger_time_opt(self):
            pts, w = self._cached_trigger_data()
            kt = self.trigger_time_provider(pts[0], w)
            return kernels.Reflect1D(kt)

        def trigger_space_opt(self):
            pts, w = self._cached_trigger_data()
            return self.trigger_space_provider(pts[1:], w)

        def iterate(self):
            return Model1(T=self.model.T, mu=self.mu_opt(), theta=self.theta_opt(),
                background_kernel=self.background_opt(),
                trigger_time_kernel=self.trigger_time_opt(),
                trigger_space_kernel=self.trigger_space_opt())


class Optimiser1SEMFactory(Optimiser1Factory):
    """As :class:`Optimiser1Factory` but uses stochastic EM algorithm.
    
    :param background_provider: Instance of :class:`KernelProvider` to use for
      estimating the background (2D) kernel.
    :param trigger_provider: Instance of :class:`KernelProvider` to use for
      estimating the trigger (3D) kernel.
    """
    def __init__(self, background_provider, trigger_time_provider, trigger_space_provider):
        super().__init__(background_provider, trigger_time_provider, trigger_space_provider)

    def __call__(self, model, points):
        opt = self._Optimiser(model, points)
        opt.background_provider = self._background_provider
        opt.trigger_time_provider = self._trigger_time_provider
        opt.trigger_space_provider = self._trigger_space_provider
        p = sepp_base.normalise_p(sepp_base.clamp_p(opt.p, self._p_cutoff))
        opt.inject_pmatrix(p)
        return opt

    class _Optimiser(Optimiser):
        def __init__(self, *args):
            super().__init__(*args)

        def _sample_points(self):
            if not hasattr(self, "_sampled_points"):
                self._sampled_points = self.sample_to_points()
            return self._sampled_points

        def background_opt(self):
            backs, _ = self._sample_points()
            return self.background_provider(backs[1:,:], None)

        def trigger_time_opt(self):
            _, trigs = self._sample_points()
            ker = self.trigger_time_provider(trigs[0], None)
            return kernels.Reflect1D(ker)

        def trigger_space_opt(self):
            _, trigs = self._sample_points()
            return self.trigger_space_provider(trigs[1:], None)

        def iterate(self):
            return Model1(T=self.model.T, mu=self.mu_opt(), theta=self.theta_opt(),
                background_kernel=self.background_opt(),
                trigger_time_kernel=self.trigger_time_opt(),
                trigger_space_kernel=self.trigger_space_opt())


class Trainer1(Trainer):
    """Training class for :class:`Model1`
    
    :param optimiser_factory: Factory to use to build the optimiser.
    """
    def __init__(self, optimiser_factory, p_cutoff=99.9,
            initial_time_scale=1, initial_space_scale=20):
        super().__init__(optimiser_factory, p_cutoff)
        self._initial_time_scale = initial_time_scale
        self._initial_space_scale = initial_space_scale

    def to_predictor(self, model):
        raise NotImplementedError()

    def initial_model(self, T, data):
        bk = kernels.GaussianBase(data[1:,:])
        bk.covariance_matrix = _np.eye(2)
        bk.bandwidth = 50
        def tk_time(t):
            # Is it okay that these are hard-wired??
            return _np.exp(-self._initial_time_scale * _np.asarray(t)) * self._initial_time_scale
            #return 0.1 * _np.exp(-0.1 * t)
        def tk_space(pts):
            pts = _np.asarray(pts)
            rr = _np.sqrt(pts[0]**2 + pts[1]**2)
            bw = 1 / self._initial_space_scale
            return bw * bw * _np.exp(-bw * rr / 2) / (2 * _np.pi)
        return Model1(T=T, mu=data.shape[-1] / T, theta=0.5,
            background_kernel=bk, trigger_time_kernel=tk_time,
            trigger_space_kernel=tk_space)
