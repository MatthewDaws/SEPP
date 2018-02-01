import pytest

import sepp.sepp_fixed as sepp_fixed

import numpy as np
import open_cp.data
import sepp.kernels
import unittest.mock as mock

def test_ExpTimeKernel():
    f = sepp_fixed.ExpTimeKernel(3)
    assert f(2) == pytest.approx(3*np.exp(-6))
    assert f(1) == pytest.approx(3*np.exp(-3))
    np.testing.assert_allclose(f([1,2]), [3*np.exp(-3), 3*np.exp(-6)])

def test_GaussianSpaceKernel():
    g = sepp_fixed.GaussianSpaceKernel(3)
    assert g([0,0]) == pytest.approx(1/(np.pi*18))
    assert g([1,0]) == pytest.approx(np.exp(-1/18)/(np.pi*18))
    np.testing.assert_allclose(g([[0,1], [0,0]]), [1/(np.pi*18), np.exp(-1/18)/(np.pi*18)])

@pytest.fixture
def grid():
    m = np.asarray([[False]*3, [True]*3])
    return open_cp.data.MaskedGrid(xsize=10, ysize=15, xoffset=100, yoffset=125, mask=m)

def test_GridModel(grid):
    tk = sepp_fixed.ExpTimeKernel(0.5)
    sk = sepp_fixed.GaussianSpaceKernel(2)
    model = sepp_fixed.GridModel([[1,2,3],[0,0,2]], 12.3, grid, 0.35, tk, sk)

    assert model.theta == pytest.approx(0.35)
    assert model.space_kernel is sk
    assert model.time_kernel is tk

    assert model.time_trigger(3) == pytest.approx(0.35*0.5*np.exp(-1.5))
    assert model.space_trigger([0,1]) == pytest.approx(sk([1,0]))
    assert model.trigger(None, [3,0,1]) == pytest.approx(0.35*0.5*np.exp(-1.5) * sk([1,0]))

    assert repr(model).startswith("GridModel(mu size")

@pytest.fixture
def model(grid):
    tk = sepp_fixed.ExpTimeKernel(0.5)
    sk = sepp_fixed.GaussianSpaceKernel(2)
    return sepp_fixed.GridModel([[1,2,3],[0,0,2]], 12.3, grid, 0.35, tk, sk)

def test_GridOptimiser(model):
    pts = np.random.random((3,10)) + np.asarray([0,100,125])[:,None]
    opt = sepp_fixed.GridOptimiser(model, pts)

    m = opt.iterate()
    assert isinstance(m, sepp_fixed.GridModel)

def test_GridTrainer(grid):
    tk = sepp_fixed.ExpTimeKernel(0.5)
    sk = sepp_fixed.GaussianSpaceKernel(2)
    trainer = sepp_fixed.GridTrainer(grid, tk, sk)

    times = [np.datetime64("2018-01-01") + np.timedelta64(1, "h")*i for i in range(10)]
    pts = open_cp.TimedPoints(times, np.random.random((2,10))+ np.asarray([100,125])[:,None])
    trainer.data = pts

    trainer.train(np.datetime64("2018-01-12"), iterations=1)


#############################################################################
# KDE background estimation
#############################################################################

@pytest.fixture
def background_kernel_mock():
    return mock.Mock()

@pytest.fixture
def kde_model(background_kernel_mock):
    tk = sepp_fixed.ExpTimeKernel(0.5)
    sk = sepp_fixed.GaussianSpaceKernel(2)
    return sepp_fixed.KDEModel(12.3, 1.2, background_kernel_mock, 1.35, tk, sk)

def test_KDEModel(kde_model, background_kernel_mock):
    model = kde_model

    assert model.T == pytest.approx(12.3)
    assert model.mu == pytest.approx(1.2)
    assert model.theta == pytest.approx(1.35)
    assert model.background_kernel is background_kernel_mock

    assert model.time_trigger(1) == pytest.approx(1.35 * 0.5 * np.exp(-0.5))
    assert model.space_trigger([0,0]) == pytest.approx(1 / (8*np.pi))

    assert repr(model).startswith("KDEModel(T=")

def test_KDEOptimiserFactory(kde_model, background_kernel_mock):
    optfac = sepp_fixed.KDEOptimiserFactory(sepp.kernels.FixedBandwidthKernelProvider(50))

    points = np.random.random((3,10)) * 10
    background_kernel_mock.return_value = 1
    opt = optfac(kde_model, points)

    model_new = opt.iterate()
    assert isinstance(model_new, sepp_fixed.KDEModel)

def test_KDEOptimiserFactoryFixedTheta(kde_model, background_kernel_mock):
    optfac = sepp_fixed.KDEOptimiserFactoryFixedTheta(sepp.kernels.FixedBandwidthKernelProvider(50))

    points = np.random.random((3,10)) * 10
    background_kernel_mock.return_value = 1
    opt = optfac(kde_model, points)

    model_new = opt.iterate()
    assert isinstance(model_new, sepp_fixed.KDEModel)
    assert model_new.theta == pytest.approx(kde_model.theta)

def test_KDETrainer():
    tk = sepp_fixed.ExpTimeKernel(0.5)
    sk = sepp_fixed.GaussianSpaceKernel(2)
    bkp = sepp.kernels.FixedBandwidthKernelProvider(50)
    trainer = sepp_fixed.KDETrainer(tk, sk, bkp)

    times = [np.datetime64("2018-01-01") + np.timedelta64(1, "h")*i for i in range(10)]
    pts = open_cp.TimedPoints(times, np.random.random((2,10))+ np.asarray([100,125])[:,None])
    trainer.data = pts

    trainer.train(np.datetime64("2018-01-12"), iterations=2)
