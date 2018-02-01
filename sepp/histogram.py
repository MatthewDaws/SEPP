"""
histogram
=========

Supports EM optimisation of histogram based KDE methods.
"""

import numpy as _np
import logging as _logging
_logger = _logging.getLogger(__name__)


class NonEdgeCorrectHistogram():
    """Solve finding :math:`f` which maximises

      :math:`\sum_{i=1}^n p_i \log f(x_i)`

    for given "weights" :math:`(p_i)` and :math:`(x_i)` with
    :math:`p_i, x_i\geq 0`.  Here :math:`f` is of the form

      :math:`f(x) = h^{-1}\alpha_r`

    where :math:`hr \leq x < h(r+1)`.  We call :math:`h` the _bandwidth_
    and assume :math:`\alpha_r=0` for :math:`r<0`.

    This instance can be evaluated at (arrays of) points returning the value
    of the histogram.

    :param weights: Array of :math:`p` weights
    :param x: Array of points to evaluate :math:`f` at; same shape as
      `weights`.
    :param bandwidth: Value `>0`
    """
    def __init__(self, weights, x, bandwidth):
        self._p = _np.asarray(weights)
        self._x = _np.asarray(x)
        self._beta = self._calculate_beta(self._p, self._x, bandwidth)
        self._alpha = self.beta / _np.sum(self.beta)
        self._bandwidth = bandwidth

    @property
    def beta(self):
        """:math:`\beta_r = \sum \{ p_i : hr \leq x_i < h(r+1) \}`."""
        return self._beta

    @property
    def weights(self):
        """The weights we formed this instance with."""
        return self._p

    @property
    def locations(self):
        """The locations we formed this instance with."""
        return self._x

    @property
    def bandwidth(self):
        """The bandwidth we formed this instance with."""
        return self._bandwidth

    @property
    def alpha(self):
        """The values of the histogram"""
        return self._alpha

    def __call__(self, x):
        x = _np.asarray(x)
        dx = _np.floor_divide(x, self._bandwidth).astype(_np.int)
        m = dx < self._alpha.shape[0]
        out = _np.empty_like(x, dtype=_np.float)
        out[m] = self._alpha[dx[m]] / self._bandwidth
        out[~m] = 0
        return out

    @staticmethod
    def _calculate_beta(p, x, bandwidth):
        """Find :math:`\beta_r = \sum \{ p_i : hr \leq x_i < h(r+1) \}`."""
        dx = _np.floor_divide(x, bandwidth).astype(_np.int)
        beta = _np.empty(_np.max(dx)+1)
        for i in range(beta.shape[0]):
            beta[i] = _np.sum(p[dx == i])
        return beta


class EdgeCorrectHistogram():
    """As :class:`NonEdgeCorrectHistogram` but now maximise
    
      :math:`:math:\sum_{i=1}^n p_i \log f(x_i)
      - \theta \sum_{i=1}^m \int_0^{T_i} f(x) \ dx`

    Again, the instance can be evaluated.

    :param weights: Array of :math:`p` weights.
    :param x: Array of points to evaluate :math:`f` at; same shape as
      `weights`.
    :param times: Array of :math:`T_i` values.
    :param bandwidth: Value `>0`.
    """
    def __init__(self, weights, x, times, bandwidth, theta):
        self._p = _np.asarray(weights)
        self._x = _np.asarray(x)
        self._t = _np.asarray(times)
        self._h = bandwidth
        self._theta = theta
        
        self._beta = NonEdgeCorrectHistogram._calculate_beta(self._p, self._x, bandwidth)
        self._gamma = self._calculate_gamma()
        if self._gamma.shape[0] < self._beta.shape[0]:
            g = _np.empty(self._beta.shape)
            g[:self._gamma.shape[0]] = self._gamma
            g[self._gamma.shape[0]:] = 0
            self._gamma = g
        self._alpha = self._calculate_alpha()

    def __call__(self, x):
        x = _np.asarray(x)
        dx = _np.floor_divide(x, self._h).astype(_np.int)
        m = dx < self._alpha.shape[0]
        out = _np.empty_like(x, dtype=_np.float)
        out[m] = self._alpha[dx[m]] / self._h
        out[~m] = 0
        return out

    def _calculate_alpha(self):
        """:math:`\alpha_r = \beta_r / (\lambda + theta h^{-1} \gamma_r)`
        normalised.
        """
        if _np.all(self._beta <= 0):
            raise ValueError("Cannot have all beta as zero.")
        lam = self._alpha_initial_binary_search()
        #over, under = self._adjusted_beta_gamma()
        #print(over)
        #print(under)
        #print(_np.sum(over/under))
        #print(lam)
        #alpha = self._alpha_func(lam)
        #print(alpha)
        #print(_np.sum(alpha))
        #raise Exception(over, under)
        while True:
            if _np.abs(self._hfunc(lam)) <= 1e-9:
                break
            lamnew = self._alpha_nr_step(lam)
            if lamnew == lam:
                _logger.error("Convergence failure in alpha step!")
                break
            lam = lamnew
        return self._alpha_func(lam)

    def _alpha_initial_binary_search(self):
        _, g = self._adjusted_beta_gamma()
        lammin = - _np.min(g)
        #lammin = - _np.min(self._gamma[:self._beta.shape[0]]) * self._theta / self._h

        lam1 = 1
        while self._hfunc(lam1) >= 0:
            lam1 += lam1
        
        lam0 = lammin * 0.99 + lam1 * 0.01
        while self._hfunc(lam0) <= 0:
            lam0 = (lammin + lam0) / 2
        
        #print("Initial guess:", lam0, lam1, self._hfunc(lam0), self._hfunc(lam1))
        #raise Exception("", self, lam0, lam1)
        while True:
            lam = (lam0 + lam1) / 2
            h = self._hfunc(lam)
            if h > 0:
                lam0 = lam
                h0 = h
                h1 = self._hfunc(lam1)
            else:
                lam1 = lam
                h1 = h
                h0 = self._hfunc(lam0)
            if h0 - h1 < 0.5: # Reasonable?
                break
            if lam1 - lam0 < 1e-8:
                raise Exception("Convergence failure...")
            #print("Current guess:", lam0, lam1, self._hfunc(lam0), self._hfunc(lam1))
        return (lam0 + lam1) / 2

    def _adjusted_beta_gamma(self):
        """Mask where beta is very small or zero."""
        mask = self._beta < 1e-10
        under = _np.ma.array(self._theta * self._gamma[:self._beta.shape[0]] / self._h, mask=mask)
        return _np.ma.array(self._beta, mask=mask), under

    def _alpha_func(self, lam):
        over, under = self._adjusted_beta_gamma()
        under = lam + under
        if _np.any(under <= 0):
            raise ValueError("Invalid lam={}, theta={}, h={}, gamma={}".format(lam, self._theta, self._h, self._gamma[:self._beta.shape[0]]), self)
        alpha = over / under
        alpha.fill_value = 0
        return alpha.filled()

    def _hfunc(self, lam):
        return _np.sum(self._alpha_func(lam)) - 1

    def _hfunc_diff(self, lam):
        over, under = self._adjusted_beta_gamma()
        under = lam + under
        return - _np.sum(over / (under * under))

    def _alpha_nr_step(self, lam):
        return lam - self._hfunc(lam) / self._hfunc_diff(lam)

    def _calculate_gamma(self):
        """Calculate

          :math:`\sum_i \int_0^{T_i} \chi_{[r*h, (r+1)*h)}`

        for each `r`.  For each `i` we have either:

            - T_i <= r*h  so integral is 0
            - T_i >= (r+1)*h so integral is h
            - otherwise integral is T_i - r*h
        
        If :math:`\max\{T_i\} / h <= r` then we always get 0
        """
        rmax = int(_np.ceil(_np.max(self._t) / self._h))
        r = _np.arange(rmax)
        integral = self._t[:,None] - r[None,:] * self._h
        integral[integral < 0] = 0
        integral[integral > self._h] = self._h
        return _np.sum(integral, axis=0)

    @property
    def alpha(self):
        """Values of the histogram."""
        return self._alpha

    @property
    def beta(self):
        """:math:`\beta_r = \sum \{ p_i : hr \leq x_i < h(r+1) \}`.
        """
        return self._beta

    @property
    def gamma(self):
        """:math:`\gamma_r = \sum_i \int_0^{T_i} \chi_{[r*h, (r+1)*h)}`"""
        return self._gamma

    @property
    def weights(self):
        """The weights we formed this instance with."""
        return self._p

    @property
    def locations(self):
        """The locations we formed this instance with."""
        return self._x

    @property
    def bandwidth(self):
        """The bandwidth we formed this instance with."""
        return self._h

    @property
    def times(self):
        """The times we formed the instance with."""
        return self._t

    @property
    def theta(self):
        """The value of theta we formed this instance with."""
        return self._theta
