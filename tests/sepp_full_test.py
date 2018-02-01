import pytest
import unittest.mock as mock

import sepp.sepp_full as sepp_full
import numpy as np
import open_cp.kernels

@pytest.fixture
def bk():
    return mock.Mock()

@pytest.fixture
def tk():
    return mock.Mock()

@pytest.fixture
def model(bk, tk):
    return sepp_full.Model(T=123.4, mu=0.6, theta=0.45, background_kernel=bk,
                            trigger_kernel=tk)

def test_Model(model, bk, tk):
    assert model.T == pytest.approx(123.4)
    assert model.mu == pytest.approx(0.6)
    assert model.theta == pytest.approx(0.45)
    assert model.background_kernel is bk
    assert model.trigger_kernel is tk
    
    bk.return_value = 1
    assert model.background([1,2,3]) == pytest.approx(model.mu)
    
    tk.return_value = 2
    assert model.trigger(None, 5) == pytest.approx(model.theta * 2)
    
    repr(model)
    
@pytest.fixture
def model1():
    def bk(pts):
        return pts[0]**2 + pts[1]**2
    def tk(pts):
        t = pts[0]
        rr = pts[1]**2 + pts[2]**2
        return np.exp(-t) * np.exp(-rr/2)
    return sepp_full.Model(T=123.4, mu=0.6, theta=0.45, background_kernel=bk,
                           trigger_kernel=tk)

def test_Optimiser(model1, bk, tk):
    class Opt(sepp_full.Optimiser):
        def __init__(self, model, points):
            super().__init__(model, points)

        def background_opt(self):
            return bk

        def trigger_opt(self):
            return tk

    pts = np.random.random((3,10))
    opt = Opt(model1, pts)
    new_model = opt.iterate()
    
    assert new_model.background_kernel is bk
    assert new_model.trigger_kernel is tk
    assert new_model.T == model1.T

@pytest.fixture
def mock_bk_prov():
    return mock.Mock()
    
@pytest.fixture
def mock_tk_prov():
    return mock.Mock()
    
@pytest.fixture
def optimiser_factory(mock_bk_prov, mock_tk_prov):
    return sepp_full.OptimiserFactory(mock_bk_prov, mock_tk_prov)
    
def test_OptimiserFactory(model1, optimiser_factory, mock_bk_prov, mock_tk_prov):
    fac = optimiser_factory
    fac.pcutoff = 99.9
    pts = np.random.random((3,10))
    opt = fac(model1, pts)
    new_model = opt.iterate()
    
    assert new_model.background_kernel is mock_bk_prov.return_value
    assert new_model.trigger_kernel.delegate is mock_tk_prov.return_value

def test_Trainer(optimiser_factory):
    trainer = sepp_full.Trainer(optimiser_factory)
    times = [np.datetime64("2016-01-01") + np.timedelta64(1, "D") * x
             for x in range(10)]
    pts = open_cp.data.TimedPoints(times, np.random.random((2,10)))
    trainer.data = pts
    
    T, data = trainer.make_data()
    model = trainer.initial_model(T, data)
    assert isinstance(model, sepp_full.Model)

def test_OptimiserSEMFactory(model1, mock_bk_prov, mock_tk_prov):
    fac = sepp_full.OptimiserSEMFactory(mock_bk_prov, mock_tk_prov) 
    fac.pcutoff = 99.9
    pts = np.random.random((3,10))
    opt = fac(model1, pts)
    new_model = opt.iterate()
    
    assert new_model.background_kernel is mock_bk_prov.return_value
    assert new_model.trigger_kernel.delegate is mock_tk_prov.return_value

@pytest.fixture
def random_timed_points():
    times = [np.datetime64("2018-01-01") + i * np.timedelta64(1, "h") for i in range(100)]
    return open_cp.data.TimedPoints(times, np.random.random((2,100))*10)

def test_initial_model_same_as_sepp_grid_space(random_timed_points):
    mock_opt_fac = mock.Mock()
    trainer = sepp_full.Trainer(mock_opt_fac)
    trainer.data = random_timed_points
    model = trainer.initial_model(*trainer.make_data(np.datetime64("2018-02-01")))

    import sepp.sepp_grid_space
    import sepp.kernels
    mask = np.zeros((10,10))
    grid = open_cp.data.MaskedGrid(xsize=10, ysize=10, xoffset=0, yoffset=0, mask=mask)
    trainer1 = sepp.sepp_grid_space.Trainer5(grid, sepp.kernels.FixedBandwidthKernelProvider(1))
    trainer1.data = random_timed_points
    model1 = trainer1.initial_model(*trainer1.make_data(np.datetime64("2018-02-01")))

    assert model.T == pytest.approx(model1.T)
    assert model.theta == pytest.approx(model1.theta)
    pts = np.random.random((3,1000))
    np.testing.assert_allclose(model.trigger_kernel(pts), model1.trigger_kernel(pts))



#############################################################################
# Trigger kernel split is time/space
#############################################################################

@pytest.fixture
def tk_two():
    return mock.Mock()

@pytest.fixture
def model_one(bk, tk, tk_two):
    return sepp_full.Model1(T=123.4, mu=0.6, theta=0.45, background_kernel=bk,
                            trigger_time_kernel=tk, trigger_space_kernel=tk_two)

def test_Model1(model_one, bk, tk, tk_two):
    model = model_one
    assert model.T == pytest.approx(123.4)
    assert model.mu == pytest.approx(0.6)
    assert model.theta == pytest.approx(0.45)
    assert model.background_kernel is bk
    assert model.trigger_time_kernel is tk
    assert model.trigger_space_kernel is tk_two
    
    bk.return_value = 1
    assert model.background([1,2,3]) == pytest.approx(model.mu)
    
    tk.return_value = 2
    tk_two.return_value=3
    assert model.trigger(None, [5,5,6]) == pytest.approx(model.theta * 6)
    
    repr(model)

def test_Optimiser1Factory(model_one, bk, tk, tk_two):
    one, two, three = mock.Mock(), mock.Mock(), mock.Mock()
    prov = sepp_full.Optimiser1Factory(one, two, three)
    prov.pcutoff = 99.9
    bk.return_value = 5
    tk.return_value = 5
    tk_two.return_value = 5
    opt = prov(model_one, np.random.random((3,10)))
    model = opt.iterate()
    
    assert model.background_kernel is one.return_value
    assert model.trigger_time_kernel._kernel is two.return_value
    assert model.trigger_space_kernel is three.return_value

def test_Optimiser1SEMFactory(model_one, bk, tk, tk_two):
    one, two, three = mock.Mock(), mock.Mock(), mock.Mock()
    prov = sepp_full.Optimiser1SEMFactory(one, two, three)
    prov.pcutoff = 99.9
    bk.return_value = 5
    tk.return_value = 5
    tk_two.return_value = 5
    opt = prov(model_one, np.random.random((3,10)))
    model = opt.iterate()
    
    assert model.background_kernel is one.return_value
    assert model.trigger_time_kernel._kernel is two.return_value
    assert model.trigger_space_kernel is three.return_value

@pytest.fixture
def mock_tk_two_prov():
    return mock.Mock()

@pytest.fixture
def optimiser_factory1(mock_bk_prov, mock_tk_prov, mock_tk_two_prov):
    return sepp_full.Optimiser1Factory(mock_bk_prov, mock_tk_prov, mock_tk_two_prov)

def test_Trainer1(optimiser_factory1):
    trainer = sepp_full.Trainer1(optimiser_factory1)
    times = [np.datetime64("2016-01-01") + np.timedelta64(1, "D") * x
             for x in range(10)]
    pts = open_cp.data.TimedPoints(times, np.random.random((2,10)))
    trainer.data = pts
    
    T, data = trainer.make_data()
    model = trainer.initial_model(T, data)
    assert isinstance(model, sepp_full.Model1)
    