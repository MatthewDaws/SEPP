"""
kernels.py
~~~~~~~~~~

Found myself repeating too much code, so separate out some general code here,
all giving kernels providers.
"""

import numpy as _np
import open_cp.kernels as kernels


def _atleast2d(points):
    points = _np.asarray(points)
    if len(points.shape) == 1:
        points = points[:,None]
    return points


class KernelProvider():
    """ABC for different KDE methods."""
    def __call__(self, pts, weights):
        raise NotImplementedError()


class _WrapWithCutoff():
    """Class to allow pickling.  
    """
    def __init__(self, ker, cutoff):
        self._ker = ker
        self._maxsq = cutoff * cutoff
        if ker.dimension == 1:
            self._func = self._one
        else:
            self._func = self._more_than_one

    def _one(self, x):
        x = _np.atleast_1d(x)
        out = _np.empty(x.shape)
        m = (x*x) <= self._maxsq
        out[m] = self._ker(x[m])
        out[~m] = 0
        return out

    def _more_than_one(self, x):
        x = _atleast2d(x)
        out = _np.empty(x.shape[-1])
        m = _np.sum(x*x, axis=0) <= self._maxsq
        out[m] = self._ker(x[:,m])
        out[~m] = 0
        return out

    def __call__(self, x):
        return self._func(x)

    def __repr__(self):
        """Bad practise in general, but we only use this when it makes
        sense."""
        return repr(self._ker)


def compute_t_marginal(ker):
    """Return the kernel resulting from integrating out all but the first
    coordinate.  Handles the case when a cutoff is applied, but does not apply
    the cutoff in the calculation!

    :param ker: Any of the kernels returned by the providers in this module.
    """
    if isinstance(ker, kernels.Reflect1D):
        return ValueError("Already a one dimensional kernel!")
    already_reflected_axis = None
    if isinstance(ker, kernels.ReflectedKernel):
        already_reflected_axis = ker.reflected_axis
        ker = ker.delegate
    if isinstance(ker, _WrapWithCutoff):
        ker = ker._ker
    k = kernels.marginalise_gaussian_kernel(ker, 2)
    k = kernels.marginalise_gaussian_kernel(k, 1)
    if already_reflected_axis == 0:
        return kernels.Reflect1D(k)
    return k

def compute_space_marginal(ker):
    """Return the kernel resulting from integrating out all but the first
    coordinate.  Handles the case when a cutoff is applied, but does not apply
    the cutoff in the calculation!

    :param ker: Any of the kernels returned by the providers in this module.
    """
    already_reflected_axis = None
    if isinstance(ker, kernels.ReflectedKernel):
        already_reflected_axis = ker.reflected_axis
        ker = ker.delegate
    if isinstance(ker, _WrapWithCutoff):
        ker = ker._ker
    k = kernels.marginalise_gaussian_kernel(ker, 0)
    if already_reflected_axis is not None and already_reflected_axis > 0:
        return kernels.ReflectedKernel(k, already_reflected_axis - 1)
    return k


class FixedBandwidthKernelProvider(KernelProvider):
    """Use a fixed bandwidth Gaussian kernel.

    :param bandwidth: The bandwidth to use
    :param scale: If not `None`, then this will the diagonal entries of the
      covariance matrix used in the KDE.  Concretely, this means that we divide
      each coordinate by the relevant entry of `rescale`.
    :param cutoff: Distance to set the kernel to be zero at; or `None` to have
      no cutoff.
    """
    def __init__(self, bandwidth, scale=None, cutoff=None):
        self._h = bandwidth
        self._scale = scale
        self._cutoff = cutoff

    def __call__(self, pts, weights):
        pts = _np.asarray(pts)
        if weights is not None:
            weights = _np.asarray(weights)
            m = (weights > 0)
            pts = pts[...,m]
            weights = weights[m]
        ker = kernels.GaussianBase(pts)
        ker.bandwidth = self._h
        if self._scale is None:
            ker.covariance_matrix = _np.eye(ker.dimension)
        else:
            ker.covariance_matrix = _np.diag(_np.atleast_1d(self._scale))
        ker.weights = weights
        if self._cutoff is None:
            return ker
        return _WrapWithCutoff(ker, self._cutoff)

    def __repr__(self):
        out = "FixedBandwidthKernelProvider(h={}".format(self._h)
        if self._scale is not None:
            out += ",scale={}".format(self._scale)
        if self._cutoff is not None:
            out += ",cutoff={}".format(self._cutoff)
        return out + ")"


class PluginKernelProvider(KernelProvider):
    """Fixed cutoff, plugin bandwidth estimator
    
    :param max_radius: Set the kernel to be identically zero at a radius beyond
      this.  Or `None` to have no limit.
    """
    def __init__(self, max_radius=None):
        self._cutoff = max_radius

    def __call__(self, pts, weights):
        pts = _np.asarray(pts)
        if weights is not None:
            weights = _np.asarray(weights)
            m = (weights > 0)
            pts = pts[...,m]
            weights = weights[m]
        ker = kernels.GaussianBase(pts)
        ker.weights = weights
        if self._cutoff is None:
            return ker
        return _WrapWithCutoff(ker, self._cutoff)

    def __repr__(self):
        out = "PluginKernelProvider"
        if self._cutoff is not None:
            out += "(cutoff={})".format(self._cutoff)
        return out


class NearestNeighbourKernelProvider(KernelProvider):
    """Use a variable bandwidth, nearest neighbour, Gaussian kernel.

    :param k: Nearest neighbours to use.
    :param cutoff: Distance to set the kernel to be zero at; or `None` to have
      no cutoff.
    """
    def __init__(self, k, cutoff=None):
        self._k = k
        self._cutoff = cutoff

    def __call__(self, pts, weights):
        pts = _np.asarray(pts)
        if weights is not None:
            weights = _np.asarray(weights)
            m = weights > 0
            pts = pts[...,m]
            weights = weights[m]
        ker = kernels.GaussianNearestNeighbour(pts, self._k)
        ker.weights = weights
        if self._cutoff is None:
            return ker
        return _WrapWithCutoff(ker, self._cutoff)

    def __repr__(self):
        out = "NearestNeighbourKernelProvider(k={}".format(self._k)
        if self._cutoff is not None:
            out += ",cutoff={}".format(self._cutoff)
        return out + ")"

