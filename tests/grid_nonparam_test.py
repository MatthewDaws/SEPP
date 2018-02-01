import pytest
import unittest.mock as mock

import sepp.grid_nonparam as grid_nonparam
import sepp.sepp_grid
import open_cp.data
import open_cp.kernels
import numpy as np
import datetime

@pytest.fixture
def non_param_model():
    mu = [[1,2,3]]
    alpha = [0.1, 1, 4, 3, 2]
    return grid_nonparam.NonParamModel(mu, 12, 2.4, 0.34, alpha)


def test_NonParamModel_getters(non_param_model):
    model = non_param_model
    assert model.T == pytest.approx(12)
    assert model.theta == pytest.approx(2.4)
    assert model.bandwidth == pytest.approx(0.34)
    np.testing.assert_allclose(model.mu, [[1,2,3]])
    np.testing.assert_allclose(model.alpha, [0.1,1,4,3,2])
    
    with pytest.raises(ValueError):
        grid_nonparam.NonParamModel([[1,2]], 12, 2.4, 0.34, [[5,4]])

def test_NonParamModel_trigger(non_param_model):
    assert non_param_model.trigger(None, 0.2) == pytest.approx(0.24/0.34)
    assert non_param_model.trigger(None, 0.34) == pytest.approx(2.4/0.34)
    assert non_param_model.trigger(None, 0.4) == pytest.approx(2.4/0.34)
    assert non_param_model.trigger(None, 0.7) == pytest.approx(4 * 2.4/0.34)
    assert non_param_model.trigger(None, 1.1) == pytest.approx(3 * 2.4/0.34)
    assert non_param_model.trigger(None, 1.5) == pytest.approx(2 * 2.4/0.34)
    assert non_param_model.trigger(None, 2) == pytest.approx(0)
    
    np.testing.assert_allclose(non_param_model.trigger(None, [0.2, 0.34, 0.4, 0.7, 1.1, 1.5, 2]), 
        np.asarray([0.24,2.4,2.4,4 * 2.4,3 * 2.4,2 * 2.4,0])/0.34)

def test_NonParamModelOpt(non_param_model):
    points = [[[],[],[]]]
    points[0][0] = [-2, -1, -0.5, -0.1]
    opt = grid_nonparam.NonParamModelOpt(non_param_model, points)

    p = opt.pmatrix((0,0))
    assert p.shape == (4,4)
    assert p[0,0] == pytest.approx(1)
    a = 1
    b = 4 * 2.4 / 0.34
    assert p[1,1] == pytest.approx(a/(a+b))
    assert p[0,1] == pytest.approx(b/(a+b))
    a = 1
    b = 1 * 2.4 / 0.34
    c = 2 * 2.4 / 0.34
    assert p[2,2] == pytest.approx(a/(a+b+c))
    assert p[1,2] == pytest.approx(b/(a+b+c))
    assert p[0,2] == pytest.approx(c/(a+b+c))
    a = 1
    b = 2.4 / 0.34
    c = 4 * 2.4 / 0.34
    d = 0
    assert p[3,3] == pytest.approx(a/(a+b+c+d))
    assert p[2,3] == pytest.approx(b/(a+b+c+d))
    assert p[1,3] == pytest.approx(c/(a+b+c+d))
    assert p[0,3] == pytest.approx(d/(a+b+c+d))

    theta = sum(p[i,j] for j in range(1,4) for i in range(j))
    g = np.asarray([3 + 0.1/0.34, 2 + 0.16/0.34, 1 + 0.32/0.34, 1, 1])
    under = np.sum(g * non_param_model.alpha)
    theta = theta / under    
    assert opt.theta_opt() == pytest.approx(theta)
    
    model = opt.optimised_model()
    assert isinstance(model, grid_nonparam.NonParamModel)
    assert model.T == pytest.approx(12)
    assert model.bandwidth == pytest.approx(0.34)
    assert model.theta == pytest.approx(theta)
    np.testing.assert_allclose(model.alpha, opt.alpha_opt())

def test_NonParamModelOptFast(non_param_model):
    points = [[[],[],[]]]
    points[0][0] = [-2, -1, -0.5, -0.1]
    opt = grid_nonparam.NonParamModelOptFast(non_param_model, points)

    p = opt.pmatrix((0,0))
    theta = sum(p[i,j] for j in range(1,4) for i in range(j)) / 4
    assert opt.theta_opt() == pytest.approx(theta)
    
    model = opt.optimised_model()
    assert isinstance(model, grid_nonparam.NonParamModel)
    assert model.T == pytest.approx(12)
    assert model.bandwidth == pytest.approx(0.34)
    assert model.theta == pytest.approx(theta)
    np.testing.assert_allclose(model.alpha, opt.alpha_opt())
    



#############################################################################
# KDE trigger
#############################################################################

@pytest.fixture
def mock_func():
    return mock.Mock()

@pytest.fixture
def kde_model(mock_func):
    mu = [[1,2,3]]
    return grid_nonparam.KDEModel(mu, 12.3, 0.45, mock_func)

def test_KDEModel(kde_model, mock_func):
    assert kde_model.theta == pytest.approx(0.45)
    assert kde_model.trigger_func is mock_func

    mock_func.return_value = 5
    assert kde_model.trigger(None, 7) == pytest.approx(5*0.45)
    mock_func.assert_called_with(7)

@pytest.fixture
def mock_func_with_exp(mock_func):
    def func(t):
        return np.exp(-t)
    mock_func.side_effect = func
    return mock_func    

def test_KDEOpt_theta(kde_model, mock_func_with_exp):
    points = [[[],[],[]]]
    points[0][0] = [-3, -2]
    opt = grid_nonparam.KDEOpt(kde_model, points, 1.2)

    p = opt.pmatrix((0,0))
    assert p.shape == (2,2)
    assert p[0,0] == pytest.approx(1)
    a = 1
    b = np.exp(-1) * 0.45
    assert p[0,1] == pytest.approx(b/(a+b))
    assert p[1,1] == pytest.approx(a/(a+b))

    theta = b/(a+b) / 2
    assert opt.theta_opt() == pytest.approx(theta)

def test_KDEOpt(kde_model, mock_func_with_exp):
    def func(t):
        return np.exp(-t)
    mock_func.side_effect = func

    points = [[[],[],[]]]
    points[0][0] = [-4, -3, -2, -1]
    opt = grid_nonparam.KDEOpt(kde_model, points, 1.2)

    f = opt.func_opt()
    assert f.kernel.bandwidth == pytest.approx(1.2)
    assert f.kernel.covariance_matrix == pytest.approx(1)

    model = opt.optimised_model()
    assert model.theta == pytest.approx(opt.theta_opt())
    
def test_KDEOptKNN(kde_model, mock_func_with_exp):
    def func(t):
        return np.exp(-t)
    mock_func.side_effect = func

    points = [[[],[],[]]]
    points[0][0] = [-4, -3, -2, -1]
    opt = grid_nonparam.KDEOptKNN(kde_model, points, 2)

    f = opt.func_opt()
    assert isinstance(f.kernel, open_cp.kernels.GaussianNearestNeighbour)
    
def test_KNNProvider(kde_model):
    points = [[[],[],[]]]
    points[0][0] = [-4, -3, -2, -1]

    prov = grid_nonparam.KDEProviderKthNearestNeighbour(5)
    opt = prov.make_opt(kde_model, points)
    assert isinstance(opt, grid_nonparam.KDEOptKNN)
    
def test_KDETrainer():
    grid = sepp.sepp_grid.ConcreteBoundedGrid(10, 10, 0, 0, 5, 2)
    pr = grid_nonparam.KDEProviderFixedBandwidth(1.2)
    trainer = grid_nonparam.KDETrainer(grid, provider=pr, timeunit=datetime.timedelta(hours=6))
    
    tps = open_cp.data.TimedPoints.from_coords(
            [datetime.datetime(2017,1,d) for d in range(1, 11)],
            np.random.random(10) * 50,
            np.random.random(10) * 20 )
    trainer.data = tps
    
    cells, model = trainer.initial_model(None)
    omega = model.trigger_func(0)
    assert omega == pytest.approx(6/24)
    assert model.theta == pytest.approx(0.5)
    
    print("Small chance this can fail, as random data...")
    model = trainer.train(iterations=1)
    assert isinstance(model, grid_nonparam.KDEModel)
    
    assert isinstance(trainer.provider, grid_nonparam.KDEProviderFixedBandwidth)
    assert trainer.provider.bandwidth == pytest.approx(1.2)
