import pytest
import unittest.mock as mock

import sepp.sepp_grid_space as sgs
import sepp.sepp_grid
import sepp.kernels

import numpy as np
import open_cp.data
import datetime

@pytest.fixture
def mu():
    return np.random.random(size=(5,10))

@pytest.fixture
def grid():
    return sepp.sepp_grid.ConcreteBoundedGrid(xsize=10, ysize=5,
       xoffset=2, yoffset=3, xextent=10, yextent=5)

@pytest.fixture
def model(mu, grid):
    return sgs.Model(mu=mu, T=100, grid=grid)
    
def test_Model(mu, model):
    assert model.T == pytest.approx(100)
    np.testing.assert_allclose(model.mu, mu)
    
    assert model.background([None, 2, 3]) == pytest.approx(mu[0, 0])
    assert model.background([None, 13, 3]) == pytest.approx(mu[0, 1])
    assert model.background([None, 3, 9]) == pytest.approx(mu[1, 0])
    
    np.testing.assert_allclose(model.background_in_space(np.asarray([[2, 13, 3], [3, 3, 9]])),
                        [mu[0,0], mu[0,1], mu[1,0]])
    
    np.testing.assert_allclose( model.background([[0,0,0], [2,13,3], [3,3,9]]),
        [mu[0,0], mu[0,1], mu[1,0]])
        
def test_Optimiser(model):
    points = np.random.random(size=(3,10))
    
    with pytest.raises(NotImplementedError):
        sgs.Optimiser(model, points)
    
def test_Trainer(grid):
    trainer = sgs.Trainer(grid)
    assert (trainer.grid.xsize, trainer.grid.ysize) == (10, 5)


#############################################################################
# Exponential decay in space, "capped" gaussian in space
#############################################################################

@pytest.fixture
def grid1():
    return sepp.sepp_grid.ConcreteBoundedGrid(xsize=2, ysize=2,
       xoffset=0, yoffset=0, xextent=10, yextent=5)

@pytest.fixture
def model1(mu, grid1):
    return sgs.Model1(mu=mu, T=12, grid=grid1, theta=0.3, omega=10.2,
                        alpha=0.001, sigmasq=144, r0=5)

def test_Model1(mu, model1):
    assert model1.T == pytest.approx(12)
    np.testing.assert_allclose(model1.mu, mu)
    assert model1.theta == pytest.approx(0.3)
    assert model1.omega == pytest.approx(10.2)
    assert model1.alpha == pytest.approx(0.001)
    assert model1.sigma == pytest.approx(12)
    assert model1.sigmasq == pytest.approx(144)
    assert model1.r0 == pytest.approx(5)
    
    beta = 1 - np.pi * 0.001 * 25
    beta /= ( 2 * np.pi * 12 * 12 * np.exp(-25/288))
    assert model1.beta == pytest.approx(beta)
    
    assert repr(model1).startswith("Model1")

def test_Model1_trigger(model1):
    assert model1.trigger(None, [0, 1, 0]) == pytest.approx(0.3 * 10.2 * 0.001)
    assert model1.trigger(None, [0, 1, 1]) == pytest.approx(0.3 * 10.2 * 0.001)
    assert model1.trigger(None, [0, 5, 1]) == pytest.approx(0.3 * 10.2 * model1.beta * np.exp(-26 / 288))
    assert model1.trigger(None, [0, 5, 3]) == pytest.approx(0.3 * 10.2 * model1.beta * np.exp(-34 / 288))
    assert model1.trigger(None, [2, 1, 0]) == pytest.approx(0.3 * 10.2 * np.exp(-20.4) * 0.001)
    assert model1.trigger(None, [3, 5, 3]) == pytest.approx(0.3 * 10.2 * np.exp(-30.6) * model1.beta * np.exp(-34 / 288))
    
    np.testing.assert_allclose(model1.trigger(None, [[0,0,0,0,2,3], [1,1,5,5,1,5], [0,1,1,3,0,3]]),
        0.3 * 10.2 * np.asarray([0.001, 0.001, model1.beta * np.exp(-26 / 288),
            model1.beta * np.exp(-34 / 288), np.exp(-20.4) * 0.001,
            np.exp(-30.6) * model1.beta * np.exp(-34 / 288)]) )
    
    assert model1.time_trigger(5) == pytest.approx(0.3 * 10.2 * np.exp(-10.2*5))
    assert model1.time_trigger(2) == pytest.approx(0.3 * 10.2 * np.exp(-10.2*2))
    np.testing.assert_allclose(model1.time_trigger(np.asarray([2,5])),
        0.3 * 10.2 * np.exp([-10.2*2, -10.2*5]))

    assert model1.space_trigger([1,0]) == pytest.approx(0.001)
    assert model1.space_trigger([1,5]) == pytest.approx(model1.beta * np.exp(-26/288))
    np.testing.assert_allclose(model1.space_trigger(np.asarray([[1,1], [0,5]])),
        [0.001, model1.beta * np.exp(-26/288)])

def test_Model1_space_trigger_mass(model1):
    assert model1.space_trigger_mass(0) == pytest.approx(0)
    assert model1.space_trigger_mass(3) == pytest.approx(np.pi*9*0.001)
    assert model1.space_trigger_mass(5) == pytest.approx(np.pi*25*0.001)
    assert model1.space_trigger_mass(6) == pytest.approx(np.pi*25*0.001
        + model1.beta * 288*np.pi * (np.exp(-25/288) - np.exp(-36/288)) )
    
@pytest.fixture
def points():
    times = np.arange(1, 11)
    xcs = np.arange(11, 1, -1)
    ycs = np.zeros_like(xcs)
    return np.asarray([times, xcs, ycs])

@pytest.fixture
def opt1(model1, points):
    return sgs.Optimiser1(model1, points)

def test_Model1_theta_opt_simple(opt1):
    pm = opt1.p
    upper_sum = sum(pm[i,j] for j in range(1, pm.shape[0]) for i in range(j) )
    assert opt1.p_upper_tri_sum == pytest.approx(upper_sum)
    assert opt1.theta_no_edge_opt() == pytest.approx(upper_sum / 10)

def test_Model1_theta_opt(opt1):
    dt = 12 - np.arange(1, 11)
    lower = 10 - np.sum(np.exp(-10.2 * dt))
    assert opt1.theta_opt() == pytest.approx(opt1.p_upper_tri_sum / lower)

def test_Model1_omega_opt(opt1):
    pm = opt1.p
    lower1 = sum(pm[i,j] * (j-i) for j in range(1, pm.shape[0]) for i in range(j) )
    dt = 12 - np.arange(1, 11)
    lower2 = 0.3 * np.sum(dt * np.exp(-10.2 * dt))
    assert opt1.omega_opt() == pytest.approx(opt1.p_upper_tri_sum / (lower1 + lower2))
    
def test_Model1_mu_opt(opt1):
    mu = np.zeros((5, 10))
    for i, (x, y) in enumerate(zip(opt1.points[1], opt1.points[2])):
        gx = int(x/2)
        gy = int(y/2)
        mu[gy, gx] += opt1.p[i,i]
    mu = mu / (12 * 4)
    
    np.testing.assert_allclose(opt1.mu_opt(opt1.model.grid), mu)
    
def test_Model1_alpha_opt(opt1):
    a, b = 0, 0
    pm = opt1.p
    for j in range(2, 10):
        for i in range(j):
            ds = opt1.points[1:,j] - opt1.points[1:,i]
            if ds[0]**2 + ds[1]**2 <= 25:
                a += pm[i, j]
            else:
                b += pm[i, j]
                
    alpha = a / ( (a+b) * np.pi * 25 )
    assert opt1.alpha_opt() == pytest.approx(alpha)
    
def test_Model1_sigmasq_opt(opt1):
    b, c = 0, 0
    pm = opt1.p
    for j in range(2, 10):
        for i in range(j):
            ds = opt1.points[1:,j] - opt1.points[1:,i]
            sq = ds[0]**2 + ds[1]**2
            if sq > 25:
                b += pm[i, j]
                c += pm[i, j] * sq
                
    sigma2 = (c - b * 25) / (b + b)
    assert opt1.sigmasq_opt() == pytest.approx(sigma2)
    
def test_Model1_iterate(opt1):
    model = opt1.iterate()
    assert isinstance(model, sgs.Model1)
    
def test_Trainer1(grid1):
    trainer = sgs.Trainer1(grid1, 5)
    times = [np.datetime64("2017-01-01") + np.timedelta64(days=i) for i in range(10)]
    xcs = np.random.random(10) * 20
    ycs = np.random.random(10) * 10
    trainer.data = open_cp.data.TimedPoints.from_coords(times, xcs, ycs)
    
    T, points = trainer.make_data(np.datetime64("2017-01-20"))
    assert T == pytest.approx(19)
    trainer.initial_model(T, points)
    
    
#############################################################################
# Histogram in time, circular window in space
#############################################################################

# Somewhat basic tests

@pytest.fixture
def model3(mu, grid):
    return sgs.Model3(mu=mu, T=123, grid=grid, theta=12.3, bandwidth=0.23, alpha=[1,2,3,4], r0=5)

def test_Model3(model3):
    model = model3
    assert model.bandwidth == pytest.approx(0.23)
    np.testing.assert_allclose(model.alpha_array, [1,2,3,4])
    
    assert model.trigger(None, [0,1,2]) == pytest.approx(model.alpha * 12.3 / 0.23)
    assert model.trigger(None, [0.24,1,2]) == pytest.approx(model.alpha * 12.3 * 2 / 0.23)
    assert model.trigger(None, [1,1,2]) == pytest.approx(0)
    np.testing.assert_allclose(model.trigger(None, [[0,0.24,1],[1]*3, [2]*3]),
                    [model.alpha*12.3 / 0.23, model.alpha*12.3*2 / 0.23, 0])

def test_Opt3(model3):
    pts = np.random.random((3,10))
    opt = sgs.Optimiser3(model3, pts)
    opt.theta_opt()
    opt.alpha_opt()
    model = opt.iterate()
    assert isinstance(model, sgs.Model3)
    


#############################################################################
# KDE is space and time, separately
#############################################################################

def test_Model4(grid):
    tkp = sepp.kernels.FixedBandwidthKernelProvider(1)
    skp = sepp.kernels.FixedBandwidthKernelProvider(1)
    trainer = sgs.Trainer4(grid, tkp, skp, 100)
    
    times = [datetime.datetime(2017,1,1) + datetime.timedelta(days=i) for i in range(10)]
    xcs = np.random.random(10) * 100
    ycs = np.random.random(10) * 25
    data = open_cp.data.TimedPoints.from_coords(times, xcs, ycs)
    trainer.data = data
    
    T, data = trainer.make_data()
    model = trainer.initial_model(T, data)
    assert isinstance(model, sgs.Model4)
    assert model.theta == pytest.approx(0.5)
    
    assert model.time_kernel(0) == pytest.approx(1)
    assert model.time_kernel(0.5) == pytest.approx(np.exp(-0.5))
    np.testing.assert_allclose(model.time_kernel([0,0.5,1]), np.exp([0,-0.5,-1]))
    
    assert model.space_kernel([0,0]) == pytest.approx(1/(800*np.pi))
    assert model.space_kernel([1,1]) == pytest.approx(np.exp(-np.sqrt(2)/20) / np.pi / 800)
    np.testing.assert_allclose(model.space_kernel([[0,1], [0,1]]),
            [1/np.pi/800, np.exp(-np.sqrt(2)/20)/np.pi/800])
    
    assert model.trigger(None, [0,0,0]) == pytest.approx(0.5 * 1 / np.pi / 800)
    
    opt = trainer._optimiser(model, data)
    assert isinstance(opt, sgs.Optimiser4)
    
    model = opt.iterate()
    assert isinstance(model, sgs.Model4)
    


#############################################################################
# KDE is space and time together
#############################################################################

def test_Model5(mu, grid):
    trig = mock.Mock()
    model = sgs.Model5(mu, 123.4, grid, 0.23, trig)

    assert model.T == pytest.approx(123.4)
    assert model.theta == pytest.approx(0.23)
    assert model.trigger_kernel is trig
    trig.return_value = 5
    assert model.trigger(None, [2]) == pytest.approx(5*0.23)
    trig.assert_called_with([2])

def test_Trainer5(grid):
    kp = sepp.kernels.FixedBandwidthKernelProvider(5)
    trainer = sgs.Trainer5(grid, kp, 100)
    
    times = [datetime.datetime(2017,1,1) + datetime.timedelta(days=i) for i in range(10)]
    xcs = np.random.random(10) * 100
    ycs = np.random.random(10) * 25
    data = open_cp.data.TimedPoints.from_coords(times, xcs, ycs)
    trainer.data = data
    
    T, data = trainer.make_data()
    model = trainer.initial_model(T, data)
    assert isinstance(model, sgs.Model5)
    assert model.theta == pytest.approx(0.5)
    assert model.trigger(None, [0,0,0]) == pytest.approx(0.5 * 1 / np.pi / 800)
    
    opt = trainer._optimiser(model, data)
    assert isinstance(opt, sgs.Optimiser5)
    
    model = opt.iterate()
    assert isinstance(model, sgs.Model5)
