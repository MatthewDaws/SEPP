import pytest

import sepp.kernels as kernels

import numpy as np
import open_cp.kernels

def test_FixedBandwidthKernelProvider():
    prov = kernels.FixedBandwidthKernelProvider(5)
    x = [1,2,3,4]
    w = [4,3,2,1]
    ker = prov(x, w)
    
    ker1 = open_cp.kernels.GaussianBase(x)
    ker1.covariance_matrix = 1
    ker1.weights = w
    ker1.bandwidth = 5
    
    assert ker(2) == pytest.approx(ker1(2))
    assert ker(0) == pytest.approx(ker1(0))
    np.testing.assert_allclose(ker([2,1,0]), ker1([2,1,0]))
    assert ker(100) > 0
    
    prov = kernels.FixedBandwidthKernelProvider(5, cutoff=10)
    x = [1,2,3,4]
    w = [4,3,2,1]
    ker = prov(x, w)
    assert ker(2) == pytest.approx(ker1(2))
    assert ker(0) == pytest.approx(ker1(0))
    np.testing.assert_allclose(ker([2,1,0]), ker1([2,1,0]))
    assert ker(100) == 0

    prov = kernels.FixedBandwidthKernelProvider(5, scale=3)
    x = [1,2,3,4]
    w = [4,3,2,1]
    ker = prov(x, w)
    ker1.covariance_matrix = 3
    assert ker(2) == pytest.approx(ker1(2))
    assert ker(0) == pytest.approx(ker1(0))
    np.testing.assert_allclose(ker([2,1,0]), ker1([2,1,0]))
    assert ker(100) > 0

def test_NearestNeighbourKernelProvider():
    prov = kernels.NearestNeighbourKernelProvider(7)
    x = np.random.random((3,20))
    w = np.random.random(20) / 2
    ker = prov(x, w)
    
    ker1 = open_cp.kernels.GaussianNearestNeighbour(x, k=7)
    ker1.weights = w
    
    assert ker([0,0,0]) == pytest.approx(ker1([0,0,0]))
    assert ker([0,2,3]) == pytest.approx(ker1([0,2,3]))
    np.testing.assert_allclose(ker([[0,0],[2,3],[1,2]]), ker1([[0,0],[2,3],[1,2]]))
    assert ker([2,5,6]) > 0
    
    prov = kernels.NearestNeighbourKernelProvider(7, cutoff = 8)
    ker = prov(x, w)
    assert ker([0,0,0]) == pytest.approx(ker1([0,0,0]))
    assert ker([0,2,3]) == pytest.approx(ker1([0,2,3]))
    assert ker([2,5,6]) == 0
    
def test_compute_t_marginal():
    x = np.random.random((3,20))
    w = np.random.random(20) / 2

    prov = kernels.FixedBandwidthKernelProvider(5, scale=[1,2,3])
    ker = prov(x, w)
    k = kernels.compute_t_marginal(ker)
    assert isinstance(k, open_cp.kernels.GaussianBase)
    k = kernels.compute_space_marginal(ker)
    assert isinstance(k, open_cp.kernels.GaussianBase)

    prov = kernels.FixedBandwidthKernelProvider(5, scale=[1,2,3], cutoff=10)
    ker = prov(x, w)
    k = kernels.compute_t_marginal(ker)
    assert isinstance(k, open_cp.kernels.GaussianBase)

    ker1 = open_cp.kernels.ReflectedKernel(ker, 1)
    k = kernels.compute_t_marginal(ker1)
    assert isinstance(k, open_cp.kernels.GaussianBase)
    
    ker1 = open_cp.kernels.ReflectedKernel(ker, 0)
    k = kernels.compute_t_marginal(ker1)
    assert isinstance(k, open_cp.kernels.Reflect1D)

def test_PluginKernelProvider():
    prov = kernels.PluginKernelProvider()
    x = np.random.random((3,20))
    w = np.random.random(20) / 2
    ker = prov(x, w)
    assert isinstance(ker, open_cp.kernels.GaussianBase)
    k = open_cp.kernels.GaussianBase(x)
    k.weights = w
    pts = np.random.random((3,1000))
    np.testing.assert_allclose(ker(pts), k(pts))
    assert ker([5.1,0,0]) > 0

    prov = kernels.PluginKernelProvider(5)
    ker = prov(x, w)
    pts = np.random.random((3,1000))
    np.testing.assert_allclose(ker(pts), k(pts))
    assert ker([5.1,0,0]) == 0
