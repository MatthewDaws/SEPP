import pytest
import unittest.mock as mock

import sepp.histogram as histogram
import numpy as np
import math

@pytest.fixture
def hist1():
    p = np.arange(6)
    x = [0.5, 1.3, 2, 2.1, 0.7, 0.8]
    return histogram.NonEdgeCorrectHistogram(p, x, 1.3)

def test_NonEdgeCorrectHistogram_beta(hist1):
    assert hist1.bandwidth == pytest.approx(1.3)
    np.testing.assert_allclose(hist1.beta, [0+4+5, 1+2+3])

def test_NonEdgeCorrectHistogram_alpha(hist1):
    np.testing.assert_allclose(hist1.alpha, [9/15, 6/15])
    
def test_NonEdgeCorrectHistogram_eval(hist1):
    assert hist1(0.5) == pytest.approx(9/15/1.3)
    assert hist1(1.29) == pytest.approx(9/15/1.3)
    assert hist1(1.3) == pytest.approx(6/15/1.3)
    assert hist1(2) == pytest.approx(6/15/1.3)
    assert hist1(27) == pytest.approx(0)

    np.testing.assert_allclose(hist1([0.5, 1.29, 1.3, 2, 27]), np.asarray([9,9,6,6,0])/(15*1.3))
    np.testing.assert_allclose(hist1([[0.5, 1.29, 1.3], [2, 27, 3]]), np.asarray([[9,9,6],[6,0,0]])/(15*1.3))

def slow_hist(p, x, h):
    psum = np.sum(p)
    k = [int(math.floor(xx/h)) for xx in x]
    def func(t):
        out = 0
        for i, (pp, kk) in enumerate(zip(p,k)):
            if h*kk <= t and t < h*(kk+1):
                out += pp
        return out / psum / h
    return func

def test_NonEdgeCorrectHistogram_eval_against_slow(hist1):
    f = slow_hist(hist1.weights, hist1.locations, hist1.bandwidth)
    for t in np.random.random(10) * 4:
        assert f(t) == pytest.approx(float(hist1(t)))

@pytest.fixture
def hist2():
    p = np.arange(6) / 5
    x = [0.2, 1.4, 1.6, 0.8, 0.7, 2]
    t = [1, 2, 2.5, 4, 0.2, 1.2]
    return histogram.EdgeCorrectHistogram(p, x, t, 1.4, 3.2)

def test_EdgeCorrectHistogram_beta(hist2):
    assert hist2.beta[0] == pytest.approx((0+3+4)/5)
    assert hist2.beta[1] == pytest.approx((1+2+5)/5)
    assert hist2.beta.shape == (2,)

def test_EdgeCorrectHistogram_alpha(hist2):
    assert np.sum(hist2.alpha) == pytest.approx(1)

def test_EdgeCorrectHistogram_properties(hist2):
    np.testing.assert_allclose(hist2.weights, np.arange(6) / 5)
    np.testing.assert_allclose(hist2.locations, [0.2, 1.4, 1.6, 0.8, 0.7, 2])
    np.testing.assert_allclose(hist2.times, [1, 2, 2.5, 4, 0.2, 1.2])
    assert hist2.bandwidth == pytest.approx(1.4)
    assert hist2.theta == pytest.approx(3.2)

def test_EdgeCorrectHistogram_gamma(hist2):
    hist2.gamma[0] == pytest.approx(1+1.4+1.4+1.4+0.2+1.2)
    hist2.gamma[1] == pytest.approx(0+0.6+1.1+1.4+0+0)
    hist2.gamma[2] == pytest.approx(0+0+0+1.2+0+0)
    assert hist2.gamma.shape == (3,)

def test_EdgeCorrectHistogram_gamma1():
    p = np.arange(3)
    x = [0.2, 1.4, 1.6]
    t = [1,2,3]
    hist = histogram.EdgeCorrectHistogram(p, x, t, 1, 3)
    assert hist.gamma.shape == (3,)
    np.testing.assert_allclose(hist.gamma, [3,2,1])

def test_EdgeCorrectHistogram_call():
    p = np.arange(3)
    x = [0.2, 1.4, 1.6]
    t = [1,2,3]
    hist = histogram.EdgeCorrectHistogram(p, x, t, bandwidth=1, theta=3)
    print(hist.alpha, hist.beta, hist.gamma)
    assert hist(0) == pytest.approx(0)
    assert hist(0.5) == pytest.approx(0)
    assert hist(1) == pytest.approx(1)
    assert hist(1.5) == pytest.approx(1)
    assert hist(2) == pytest.approx(0)
    assert hist(3) == pytest.approx(0)
    np.testing.assert_allclose(hist([0, 0.5, 1, 1.5, 2, 3]), [0,0,1,1,0,0])

def test_EdgeCorrectHistogram_2cell_opt():
    size = 20
    for _ in range(10):
        p = np.random.random(size)
        x = np.random.random(size) * 2
        t = np.random.random(size) * 3
        theta = np.random.random() * 10
        hist = histogram.EdgeCorrectHistogram(p, x, t, bandwidth=1, theta=theta)
    
        p0 = sum(pp for pp, xx in zip(p, x) if xx < 1)
        p1 = sum(pp for pp, xx in zip(p, x) if 1 <= xx and xx < 2)
        b0, b1 = 0, 0
        for tt in t:
            if tt < 1:
                b0 += tt
            elif tt < 2:
                b0 += 1
                b1 += tt - 1
            else:
                b0 += 1
                b1 += 1
        print(p0, p1, b0, b1)
        gamma = theta * (b0 - b1)
    
        dis = p0 + p1 + gamma
        roots = []
        for si in [1, -1]:
            root = (dis + si * np.sqrt(dis * dis - 4 * gamma * p0)) / (2 * gamma)
            if 0 < root and root < 1:
                roots.append(root)
        values = [p0 * np.log(r) + p1 * np.log(1 - r) - gamma * r - theta * b1
            for r in roots]
        v = min(values)
        i = values.index(v)
        alpha = roots[i]
    
        np.testing.assert_allclose(hist.alpha, [alpha, 1-alpha])
    
    
    