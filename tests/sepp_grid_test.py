import pytest
import unittest.mock as mock

import sepp.sepp_grid as sepp_grid

import open_cp.data
import numpy as np
import datetime

def test_ConcreteBoundedGrid():
    grid = sepp_grid.ConcreteBoundedGrid(10, 12, 5, 9, 32, 25)
    assert (grid.xsize, grid.ysize) == (10, 12)
    assert (grid.xoffset, grid.yoffset) == (5, 9)
    assert (grid.xextent, grid.yextent) == (32, 25)

@pytest.fixture
def grid1():
    mat = np.random.random(size=(15,7))
    return open_cp.data.MaskedGrid(10, 20, 5, 7, mat)

@pytest.fixture
def trainer1(grid1):
    return sepp_grid.SEPPGridTrainer(grid1)

def test_SEPPGridTrainer_constructs(trainer1):
    assert trainer1.time_unit / np.timedelta64(1, "h") == 24
    assert trainer1.grid.xsize == 10
    assert trainer1.grid.ysize == 20
    assert trainer1.grid.xoffset == 5
    assert trainer1.grid.yoffset == 7
    assert trainer1.grid.xextent == 7
    assert trainer1.grid.yextent == 15

def test_SEPPGridTrainer_setters(trainer1):
    trainer1.time_unit = datetime.timedelta(hours=2)
    assert trainer1.time_unit / np.timedelta64(1, "h") == 2
    trainer1.time_unit = np.timedelta64(30, "m")
    assert trainer1.time_unit / np.timedelta64(1, "h") == pytest.approx(0.5)

    grid = mock.Mock()
    grid.xsize = 15
    grid.ysize = 25
    grid.xoffset = -5
    grid.yoffset = -7
    grid.xextent = 3
    grid.yextent = 5
    trainer1.grid = grid
    assert (trainer1.grid.xsize, trainer1.grid.ysize) == (15, 25)
    assert (trainer1.grid.xoffset, trainer1.grid.yoffset) == (-5, -7)
    assert (trainer1.grid.xextent, trainer1.grid.yextent) == (3, 5)

@pytest.fixture
def timed_points1():
    d = datetime.datetime(2017,4,5)
    hour = datetime.timedelta(hours=1)
    times = [d + k * hour for k in range(7)]
    xcs = [5, 15, 7, 17, 27, 28, 29]
    ycs = [7, 27, 7, 27, 7, 27, 7]
    return open_cp.data.TimedPoints.from_coords(times, xcs, ycs)

def test_SEPPGridTrainer_to_cells(trainer1, timed_points1):
    trainer1.data = timed_points1
    cells = trainer1.to_cells(datetime.datetime(2017,4,6))

    assert cells.shape == (15, 7)
    np.testing.assert_allclose(cells[0,0], [-24/24, -22/24])
    np.testing.assert_allclose(cells[0,2], [-20/24, -18/24])
    np.testing.assert_allclose(cells[1,1], [-23/24, -21/24])
    np.testing.assert_allclose(cells[1,2], [-19/24])
    np.testing.assert_allclose(cells[0,1], [])
    np.testing.assert_allclose(cells[1,0], [])

def test_ModelBase():
    mu = np.random.random((5,7))
    model = sepp_grid.ModelBase(mu, 12.5)
    np.testing.assert_allclose(model.mu, mu)
    assert model.T == pytest.approx(12.5)
    with pytest.raises(NotImplementedError):
        model.trigger(None, 2)

def test_prediction_from_background(trainer1):
    mu = np.random.random((5,7))
    model = sepp_grid.ModelBase(mu, 12.5)
    pred = trainer1.prediction_from_background(model)

    assert (pred.xsize, pred.ysize) == (10, 20)
    assert (pred.xoffset, pred.yoffset) == (5, 7)
    assert (pred.xextent, pred.yextent) == (7, 5)
    np.testing.assert_allclose(mu, pred.intensity_matrix)


class OurModel(sepp_grid.ModelBase):
    def trigger(self, cell, tdelta):
        return np.exp(-tdelta)


def test_prediction(trainer1):
    mu = np.random.random((5,7))
    model = OurModel(mu, 12.5)
    points = [ [np.array([]) for _ in range(7)] for _ in range(5) ]
    points[0][0] = np.asarray([-1, -2, -3])
    points = np.asarray(points)
    assert points.shape == mu.shape
    pred = trainer1.prediction(model, points, 0.5, 2)

    assert (pred.xsize, pred.ysize) == (10, 20)
    assert (pred.xoffset, pred.yoffset) == (5, 7)
    assert (pred.xextent, pred.yextent) == (7, 5)
    t = np.linspace(0.5, 2, 100)
    mu[0,0] += sum(np.sum(np.exp(-s-1) + np.exp(-s-2) + np.exp(-s-3)) for s in t) / len(t)
    np.testing.assert_allclose(mu, pred.intensity_matrix)

@pytest.fixture
def our_model1():
    mu = np.random.random((2,3))
    return OurModel(mu, 12.5)

@pytest.fixture
def points1():
    points = [ [ [], [], [] ], [ [], [], [] ] ]
    points[0][0] = [-7, -3, -2]
    points[1][0] = [-4, -2]
    return points

@pytest.fixture
def optimiser_base1(our_model1, points1):
    return sepp_grid.OptimiserBase(our_model1, points1)

def test_OptimiserBase(our_model1, optimiser_base1):
    assert optimiser_base1.T == pytest.approx(12.5)
    np.testing.assert_allclose(our_model1.mu, optimiser_base1.mu)
    expected = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    assert list(optimiser_base1.cell_iter()) == expected

    p = optimiser_base1.pmatrix((0,0))
    m = our_model1.mu[0,0]
    expected = [[m, np.exp(-4), np.exp(-5)], 
                [0, m, np.exp(-1)],
                [0, 0, m]  ]
    for j in range(3):
        n = sum(expected[i][j] for i in range(3))
        for i in range(3):
            expected[i][j] /= n
    np.testing.assert_allclose(p, expected)

    p = optimiser_base1.pmatrix((1,0))
    m = our_model1.mu[1,0]
    expected = [[m, np.exp(-2)],
                [0, m] ] 
    for j in range(2):
        n = sum(expected[i][j] for i in range(2))
        for i in range(2):
            expected[i][j] /= n
    np.testing.assert_allclose(p, expected)

    p = optimiser_base1.pmatrix((1,1))
    assert p.shape == (0, 0)

def test_OptimiserBase_mu_opt(our_model1, optimiser_base1):
    mu = optimiser_base1.mu_opt()

    m = our_model1.mu[0,0]
    expected = [[m, np.exp(-4), np.exp(-5)], 
                [0, m, np.exp(-1)],
                [0, 0, m]  ]
    for j in range(3):
        n = sum(expected[i][j] for i in range(3))
        for i in range(3):
            expected[i][j] /= n
    assert optimiser_base1.p_upper_tri_sum((0,0)) == pytest.approx(expected[0][1] + expected[0][2] + expected[1][2])
    expected = sum(expected[i][i] for i in range(3)) / 12.5
    assert mu[0,0] == pytest.approx(expected)

    m = our_model1.mu[1,0]
    expected = [[m, np.exp(-2)],
                [0, m] ] 
    for j in range(2):
        n = sum(expected[i][j] for i in range(2))
        for i in range(2):
            expected[i][j] /= n
    expected = sum(expected[i][i] for i in range(2)) / 12.5
    assert mu[1,0] == pytest.approx(expected)

    assert np.all(np.abs(mu[:, 1:]) < 1e-16)

@pytest.fixture
def optimiser_base2(our_model1):
    points = [ [ [], [], [] ], [ [], [], [] ] ]
    points[0][0] = [-3, -3, -2]
    points[1][0] = [-4, -2]
    return sepp_grid.OptimiserBaseWithRepeats(our_model1, points)

def test_OptimiserBaseWithRepeats(our_model1, optimiser_base2):
    p = optimiser_base2.pmatrix((0,0))
    m = our_model1.mu[0,0]
    expected = [[m, 0, np.exp(-1)], 
                [0, m, np.exp(-1)],
                [0, 0, m]  ]
    for j in range(3):
        n = sum(expected[i][j] for i in range(3))
        for i in range(3):
            expected[i][j] /= n
    np.testing.assert_allclose(p, expected)


#############################################################################
# Exp model stuff
#############################################################################

@pytest.fixture
def exp_decay_model1():
    mu = [[2, 3, 4], [1, 0.5, 2.5]]
    return sepp_grid.ExpDecayModel(mu, 10, 0.23, 4)

def test_ExpDecayModel_constructs(exp_decay_model1):
    assert exp_decay_model1.theta == pytest.approx(0.23)
    assert exp_decay_model1.omega == pytest.approx(4)
    assert exp_decay_model1.T == pytest.approx(10)
    np.testing.assert_allclose(exp_decay_model1.mu, [[2, 3, 4], [1, 0.5, 2.5]])
    assert exp_decay_model1.trigger(None, 3) == pytest.approx(4 * 0.23 * np.exp(-4*3))

def test_ExpDecayModel_likelihood(exp_decay_model1):
    points = [ [ [], [], [] ], [ [], [], [] ] ]
    points[0][0] = [-7, -3, -1]
    
    got = exp_decay_model1.log_likelihood(points)
    expected = np.log(2) + np.log(2 + 0.23*4*np.exp(-4*4))
    expected += np.log(2 + 0.23*4*np.exp(-4*5) + 0.23*4*np.exp(-4*2))
    expected -= 13*10
    expected -= 0.23 * ( (1 - np.exp(-4)) + (1 - np.exp(-4*3)) + (1 - np.exp(-4*7)) )
    assert got == pytest.approx(expected)

@pytest.fixture
def exp_decay_opt1(exp_decay_model1):
    points = [ [ [], [], [] ], [ [], [], [] ] ]
    points[0][0] = [-7, -3, -2]
    points[1][0] = [-4, -2]
    return sepp_grid.ExpDecayOptFast(exp_decay_model1, points)

def test_ExpDecayOptFast(exp_decay_opt1):
    assert exp_decay_opt1.total_event_count == 5

    p = exp_decay_opt1.pmatrix((0,0))
    assert p.shape == (3,3)
    pp = exp_decay_opt1.pmatrix((1,0))
    assert pp.shape == (2,2)
    theta = (p[0,1] + p[0,2] + p[1,2] + pp[0,1]) / 5
    assert exp_decay_opt1.theta_opt() == pytest.approx(theta)

    omega = p[0,1] + p[0,2] + p[1,2] + pp[0,1]
    omega /= (p[0,1] * 4 + p[0,2] * 5 + p[1,2] * 1 + pp[0,1] * 2)
    assert exp_decay_opt1.omega_opt() == pytest.approx(omega)

    model = exp_decay_opt1.optimised_model()
    np.testing.assert_allclose(model.mu, exp_decay_opt1.mu_opt())
    assert model.T == pytest.approx(10)
    assert model.theta == pytest.approx(theta)
    assert model.omega == pytest.approx(omega)

@pytest.fixture
def exp_decay_opt2(exp_decay_model1):
    points = [ [ [], [], [] ], [ [], [], [] ] ]
    points[0][0] = [-7, -3, -2]
    points[1][0] = [-4, -2]
    return sepp_grid.ExpDecayOpt(exp_decay_model1, points)

def test_ExpDecayOpt(exp_decay_opt2):
    under = np.sum(1 - np.exp(exp_decay_opt2.points[(0,0)] * 4))
    under += np.sum(1 - np.exp(exp_decay_opt2.points[(1,0)] * 4))
    p = exp_decay_opt2.pmatrix((0,0))
    assert p.shape == (3,3)
    pp = exp_decay_opt2.pmatrix((1,0))
    assert pp.shape == (2,2)
    theta = (p[0,1] + p[0,2] + p[1,2] + pp[0,1]) / under
    assert exp_decay_opt2.theta_opt() == pytest.approx(theta)

    under = (p[0,1] * 4 + p[0,2] * 5 + p[1,2] * 1 + pp[0,1] * 2)
    for tt in exp_decay_opt2.points[(0,0)]:
        under += 0.23 * (-tt) * np.exp(4 * tt)
    for tt in exp_decay_opt2.points[(1,0)]:
        under += 0.23 * (-tt) * np.exp(4 * tt)
    omega = (p[0,1] + p[0,2] + p[1,2] + pp[0,1]) / under
    assert exp_decay_opt2.omega_opt() == pytest.approx(omega)

    model = exp_decay_opt2.optimised_model()
    np.testing.assert_allclose(model.mu, exp_decay_opt2.mu_opt())
    assert model.T == pytest.approx(10)
    assert model.theta == pytest.approx(theta)
    assert model.omega == pytest.approx(omega)

@pytest.fixture
def timed_points2():
    times = [datetime.datetime(2017,1,1) + datetime.timedelta(days=365) * x
        for x in np.random.random(100)]
    xcs = np.random.random(100) * 100
    ycs = np.random.random(100) * 100
    return open_cp.data.TimedPoints.from_coords(times, xcs, ycs)

def test_ExpDecayTrainer(timed_points2):
    mask = [[False] * 5 for _ in range(5)]
    grid = open_cp.data.MaskedGrid(20, 20, 0, 0, mask)
    trainer = sepp_grid.ExpDecayTrainer(grid)
    trainer.data = timed_points2

    points, model = trainer.initial_model(datetime.datetime(2018,1,1))
    assert model.omega == pytest.approx(1)
    assert model.theta == pytest.approx(0.5)
    expected = (np.datetime64("2018-01-01") - timed_points2.time_range[0]) / np.timedelta64(1, "D")
    assert model.T == expected
    counts = [[0,0,0,0,0] for _ in range(5)]
    for x, y in zip(timed_points2.xcoords, timed_points2.ycoords):
        x, y = int(np.floor(x / 20)), int(np.floor(y / 20))
        counts[y][x] += 1
    np.testing.assert_allclose(model.mu, np.asarray(counts) / expected)

    model = trainer.train()

def test_ExpDecayTrainer_cutoff(timed_points2):
    mask = [[False] * 5 for _ in range(5)]
    grid = open_cp.data.MaskedGrid(20, 20, 0, 0, mask)
    trainer = sepp_grid.ExpDecayTrainer(grid)
    trainer.data = timed_points2

    points, model = trainer.initial_model(datetime.datetime(2017,3,1))
    assert model.omega == pytest.approx(1)
    assert model.theta == pytest.approx(0.5)
    expected = (np.datetime64("2017-03-01") - timed_points2.time_range[0]) / np.timedelta64(1, "D")
    assert model.T == expected
    counts = [[0,0,0,0,0] for _ in range(5)]
    for t, x, y in zip(timed_points2.timestamps, timed_points2.xcoords, timed_points2.ycoords):
        if t > np.datetime64("2017-03-01"):
            continue
        x, y = int(np.floor(x / 20)), int(np.floor(y / 20))
        counts[y][x] += 1
    np.testing.assert_allclose(model.mu, np.asarray(counts) / expected)


#############################################################################
# Exp model stuff with cutoff
#############################################################################

@pytest.fixture
def exp_decay_model_with_cutoff():
    mu = np.asarray([[1,2,3]])
    return sepp_grid.ExpDecayModelWithCutoff(mu, 10, 1.2, 2.3, 12)

def test_ExpDecayModelWithCutoff_trigger(exp_decay_model_with_cutoff):
    model = exp_decay_model_with_cutoff
    assert model.theta == pytest.approx(1.2)
    assert model.omega == pytest.approx(2.3)
    assert model.cutoff == pytest.approx(12)
    assert model.T == pytest.approx(10)
    np.testing.assert_allclose(model.mu, [[1,2,3]])
    assert model.trigger(None, 11.9) == pytest.approx(0)
    assert model.trigger(None, 12) == pytest.approx(1.2 * 2.3)
    assert model.trigger(None, 13) == pytest.approx(1.2 * 2.3 * np.exp(-2.3))
    np.testing.assert_allclose(model.trigger(None, [11.9, 12, 13]), [0, 1.2*2.3, 1.2 * 2.3 * np.exp(-2.3)])

    assert model.trigger_integral(5) == pytest.approx(0)
    assert model.trigger_integral(13) == pytest.approx(1 - np.exp(-2.3))
    assert model.trigger_integral(12.5) == pytest.approx(1 - np.exp(-1.15))
    np.testing.assert_allclose(model.trigger_integral([13, 5, 12.5]), [1-np.exp(-2.3), 0, 1-np.exp(-1.15)])

def test_ExpDecayModelWithCutoff_likelihood(exp_decay_model_with_cutoff):
    model = exp_decay_model_with_cutoff

    pts = [[-13, -7], [], []]
    ll = -6*10
    ll -= 1.2 * (1 - np.exp(-2.3))
    ll += np.log(1) * 2
    assert exp_decay_model_with_cutoff.log_likelihood(pts) == pytest.approx(ll)

    pts = [[-26, -13, -7], [], []]
    ll = np.log(1) * 2
    ll = np.log(1 + 1.2 * 2.3 * np.exp(-2.3 * 1) + 1.2 * 2.3 * np.exp(-2.3 * 7))
    ll += -6*10
    ll -= 1.2 * (1 - np.exp(-2.3) + 1 - np.exp(-13*2.3))
    assert exp_decay_model_with_cutoff.log_likelihood(pts) == pytest.approx(ll)

@pytest.fixture
def exp_decay_opt_cutoff_fast():
    mu = np.asarray([[1,2,3]])
    model = sepp_grid.ExpDecayModelWithCutoff(mu, T=10, theta=1.2, omega=2.3, cutoff=1.1)
    points = [[ [], [], [] ]]
    points[0][0] = [-6, -2, -1]
    points[0][1] = [-4, -2]
    return sepp_grid.ExpDecayOptFastWithCutoff(model, points)

def test_ExpDecayModelWithCutoff_fast_opt(exp_decay_opt_cutoff_fast):
    opt = exp_decay_opt_cutoff_fast
    p = opt.pmatrix((0,0))
    assert p.shape == (3,3)
    assert p[0,0] == pytest.approx(1)
    a = 1
    b = 1.2*2.3*np.exp(-2.3*2.9)
    assert p[1,1] == pytest.approx(a / (a+b))
    assert p[0,1] == pytest.approx(b / (a+b))
    a = 1
    b = 0
    c = 1.2*2.3*np.exp(-2.3*3.9)
    assert p[2,2] == pytest.approx(a / (a+b+c))
    assert p[1,2] == pytest.approx(b / (a+b+c))
    assert p[0,2] == pytest.approx(c / (a+b+c))
    
    pp = opt.pmatrix((0,1))
    assert pp.shape == (2,2)
    assert pp[0,0] == pytest.approx(1)
    a = 2
    b = 1.2*2.3*np.exp(-2.3*0.9)
    assert pp[1,1] == pytest.approx(a / (a+b))
    assert pp[0,1] == pytest.approx(b / (a+b))

    over = p[0,1] + p[0,2] + p[1,2] + pp[0,1]
    theta = over / 4
    assert opt.theta_opt() == pytest.approx(theta)

    under = p[0,1] * 2.9 + p[0,2] * 3.9 + pp[0,1] * 0.9
    omega = over / under
    assert opt.omega_opt() == pytest.approx(omega)

    mu = [(p[0,0] + p[1,1] + p[2,2]) / 10, (pp[0,0] + pp[1,1]) / 10, 0]

    model = opt.optimised_model()
    assert isinstance(model, sepp_grid.ExpDecayModelWithCutoff)
    assert model.T == pytest.approx(10)
    assert model.theta == pytest.approx(theta)
    assert model.omega == pytest.approx(omega)
    assert model.cutoff == pytest.approx(1.1)
    np.testing.assert_allclose(model.mu, [mu])

@pytest.fixture
def exp_decay_opt_cutoff():
    mu = np.asarray([[1,2,3]])
    model = sepp_grid.ExpDecayModelWithCutoff(mu, T=10, theta=1.2, omega=2.3, cutoff=1.1)
    points = [[ [], [], [] ]]
    points[0][0] = [-6, -2, -1]
    points[0][1] = [-4, -2]
    return sepp_grid.ExpDecayOptWithCutoff(model, points)

def test_ExpDecayModelWithCutoff_opt(exp_decay_opt_cutoff):
    opt = exp_decay_opt_cutoff
    p = opt.pmatrix((0,0))
    pp = opt.pmatrix((0,1))

    over = p[0,1] + p[0,2] + p[1,2] + pp[0,1]
    under = 1 - np.exp(-2.3*0.9) + 1 - np.exp(-2.3*4.9) + 1 - np.exp(-2.3*0.9) + 1 - np.exp(-2.3*2.9)
    theta = over / under
    assert opt.theta_opt() == pytest.approx(theta)

    under = p[0,1] * 2.9 + p[0,2] * 3.9 + pp[0,1] * 0.9
    under += 1.2 * ( 0.9*np.exp(-2.3*0.9) + 4.9*np.exp(-2.3*4.9) + 0.9*np.exp(-2.3*0.9) + 2.9*np.exp(-2.3*2.9) )
    omega = over / under
    assert opt.omega_opt() == pytest.approx(omega)

    mu = [(p[0,0] + p[1,1] + p[2,2]) / 10, (pp[0,0] + pp[1,1]) / 10, 0]

    model = opt.optimised_model()
    assert isinstance(model, sepp_grid.ExpDecayModelWithCutoff)
    assert model.T == pytest.approx(10)
    assert model.theta == pytest.approx(theta)
    assert model.omega == pytest.approx(omega)
    assert model.cutoff == pytest.approx(1.1)
    np.testing.assert_allclose(model.mu, [mu])
