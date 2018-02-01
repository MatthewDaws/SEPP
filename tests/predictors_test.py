import pytest
import unittest.mock as mock

import sepp.predictors as predictors
import sepp.sepp_full

import datetime

@mock.patch("sepp.sepp_grid.ExpDecayTrainer")
def test_ExpDecayGridProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    provider_factory = predictors.ExpDecayGridProvider(grid,
            points, datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, allow_repeats=True)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value

    provider = provider_factory(points, grid)
    assert repr(provider) == "ExpDecayGridProvider"
    points2 = mock.Mock()
    cells = mock.Mock()
    trainer.make_points.return_value = (cells, 10)
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5))
    assert pred is trainer.prediction.return_value
    assert trainer.data is points2
    trainer.prediction.assert_called_with(mock_model, cells)

@mock.patch("sepp.grid_nonparam.NonParamTrainer")
def test_NonParamGridProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    provider_factory = predictors.NonParamGridProvider(grid,
            points, 1.23, datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, bandwidth=1.23)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25, use_fast=True)
    mock_model = trainer.train.return_value

    provider = provider_factory(points, grid)
    assert repr(provider) == "NonParamGridProvider(h=1.23)"
    points2 = mock.Mock()
    cells = mock.Mock()
    trainer.make_points.return_value = (cells, 10)
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5))
    assert pred is trainer.prediction.return_value
    assert trainer.data is points2
    trainer.prediction.assert_called_with(mock_model, cells)

@mock.patch("sepp.grid_nonparam.KDETrainer")
def test_KDEGridProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    kde_provider = mock.Mock()
    provider_factory = predictors.KDEGridProvider(grid,
            points, kde_provider, datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, kde_provider)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("KDEGridProvider")
    points2 = mock.Mock()
    cells = mock.Mock()
    trainer.make_points.return_value = (cells, 10)
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5))
    assert pred is trainer.prediction.return_value
    assert trainer.data is points2
    trainer.prediction.assert_called_with(mock_model, cells)

@mock.patch("sepp.sepp_grid_space.Trainer1")
def test_GridSpaceExpDecayProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    provider_factory = predictors.GridSpaceExpDecayProvider(grid,
            points, 3, datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, 3)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    trainer.to_predictor.assert_called_with(mock_model)
    mock_predictor = trainer.to_predictor.return_value.to_fast_split_predictor.return_value

    provider = provider_factory(points, grid)
    assert repr(provider) == "GridSpaceExpDecayProvider(r0=3)"
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor.predict.return_value
    assert mock_predictor.data is points2
    mock_predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("sepp.sepp_grid_space.Trainer2")
def test_GridSpaceSimpleProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    provider_factory = predictors.GridSpaceSimpleProvider(grid,
            points, 3, datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, 3)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    trainer.to_predictor.assert_called_with(mock_model)
    mock_predictor = trainer.to_predictor.return_value.to_fast_split_predictor.return_value

    provider = provider_factory(points, grid)
    assert repr(provider) == "GridSpaceSimpleProvider(r0=3)"
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor.predict.return_value
    assert mock_predictor.data is points2
    mock_predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("sepp.sepp_grid_space.Trainer3")
def test_GridSpaceSimpleHistogramProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    provider_factory = predictors.GridSpaceSimpleHistogramProvider(grid,
            points, 3, 5, datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, r0=3, bandwidth=5, use_fast=True)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    trainer.to_predictor.assert_called_with(mock_model)
    mock_predictor = trainer.to_predictor.return_value.to_fast_split_predictor.return_value

    provider = provider_factory(points, grid)
    assert repr(provider) == "GridSpaceSimpleHistogramProvider(r0=3, h=5)"
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor.predict.return_value
    assert mock_predictor.data is points2
    mock_predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("sepp.sepp_grid_space.Trainer4")
def test_GridSpaceKDEProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    time_kernel_provider_mock = mock.Mock()
    space_kernel_provider_mock = mock.Mock()
    provider_factory = predictors.GridSpaceKDEProvider(grid, points,
            time_kernel_provider_mock, space_kernel_provider_mock,
            datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, time_kernel_provider_mock,
            space_kernel_provider_mock, p_cutoff=99.99)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    trainer.to_predictor.assert_called_with(mock_model)
    mock_predictor = trainer.to_predictor.return_value.to_fast_split_predictor_histogram.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("GridSpaceKDEProvider(tker=")
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor.predict.return_value
    assert mock_predictor.data is points2
    mock_predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("open_cp.sepp_base.Predictor")
@mock.patch("sepp.sepp_full.Trainer1")
def test_FullDKEProvider(mock_trainer, mock_predictor):
    grid = mock.Mock()
    points = mock.Mock()
    time_kernel_provider_mock = mock.Mock()
    space_kernel_provider_mock = mock.Mock()
    back_kernel_provider_mock = mock.Mock()
    provider_factory = predictors.FullKDEProvider(grid, points,
            time_kernel_provider_mock, space_kernel_provider_mock,
            back_kernel_provider_mock, datetime.datetime(2016,10,1), iterations=25)

    fac = mock_trainer.call_args[0][0]
    assert isinstance(fac, sepp.sepp_full.Optimiser1Factory)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    mock_predictor.assert_called_with(grid, mock_model)
    mock_predictor.return_value.to_fast_split_predictor_histogram.assert_called_with(time_bin_size=0.1, space_bin_size=10)
    mock_predictor_formed = mock_predictor.return_value.to_fast_split_predictor_histogram.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("FullKDEProvider(tker=")
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor_formed.predict.return_value
    assert mock_predictor_formed.data is points2
    mock_predictor_formed.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("sepp.sepp_fixed.GridTrainer")
def test_GridFixedTriggerProvider(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    time_kernel_mock = mock.Mock()
    space_kernel_mock = mock.Mock()
    provider_factory = predictors.GridFixedTriggerProvider(grid, points,
            time_kernel_mock, space_kernel_mock,
            cutoff=datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, time_kernel_mock, space_kernel_mock)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    trainer.to_predictor.assert_called_with(mock_model)
    mock_predictor = trainer.to_predictor.return_value.to_fast_split_predictor.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("GridFixedTriggerProvider(tker=")
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor.predict.return_value
    assert mock_predictor.data is points2
    mock_predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("sepp.sepp_fixed.GridTrainerFixedTheta")
def test_GridFixedTriggerProvider_with_theta(mock_trainer):
    grid = mock.Mock()
    points = mock.Mock()
    time_kernel_mock = mock.Mock()
    space_kernel_mock = mock.Mock()
    provider_factory = predictors.GridFixedTriggerProvider(grid, points,
            time_kernel_mock, space_kernel_mock, theta=0.2,
            cutoff=datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(grid, time_kernel_mock, space_kernel_mock, 0.2)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value
    trainer.to_predictor.assert_called_with(mock_model)
    mock_predictor = trainer.to_predictor.return_value.to_fast_split_predictor.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("GridFixedTriggerProvider(tker=")
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    assert pred is mock_predictor.predict.return_value
    assert mock_predictor.data is points2
    mock_predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)

@mock.patch("open_cp.sepp_base.Predictor")
@mock.patch("sepp.sepp_fixed.KDETrainer")
def test_KDEFixedTriggerProvider(mock_trainer, mock_predictor):
    grid = mock.Mock()
    points = mock.Mock()
    time_kernel_mock = mock.Mock()
    space_kernel_mock = mock.Mock()
    background_kp_mock = mock.Mock()
    provider_factory = predictors.KDEFixedTriggerProvider(points,
            time_kernel_mock, space_kernel_mock, background_kp_mock,
            cutoff=datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(time_kernel_mock, space_kernel_mock, background_kp_mock)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("KDEFixedTriggerProvider(tker=")
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    mock_predictor.assert_called_with(grid, mock_model)
    predictor = mock_predictor.return_value.to_fast_split_predictor.return_value
    assert predictor.data is points2
    predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)
    assert pred is predictor.predict.return_value

@mock.patch("open_cp.sepp_base.Predictor")
@mock.patch("sepp.sepp_fixed.KDETrainerFixedTheta")
def test_KDEFixedTriggerProvider_with_theta(mock_trainer, mock_predictor):
    grid = mock.Mock()
    points = mock.Mock()
    time_kernel_mock = mock.Mock()
    space_kernel_mock = mock.Mock()
    background_kp_mock = mock.Mock()
    provider_factory = predictors.KDEFixedTriggerProvider(points,
            time_kernel_mock, space_kernel_mock, background_kp_mock,
            theta=0.2, cutoff=datetime.datetime(2016,10,1), iterations=25)

    mock_trainer.assert_called_with(time_kernel_mock, space_kernel_mock, background_kp_mock, 0.2)
    trainer = mock_trainer.return_value
    assert trainer.data is points
    trainer.train.assert_called_with(datetime.datetime(2016,10,1), 25)
    mock_model = trainer.train.return_value

    provider = provider_factory(points, grid)
    assert repr(provider).startswith("KDEFixedTriggerProvider(tker=")
    points2 = mock.Mock()
    pred = provider.give_prediction(grid, points2, datetime.datetime(2016,10,5), datetime.datetime(2016,10,7))
    mock_predictor.assert_called_with(grid, mock_model)
    predictor = mock_predictor.return_value.to_fast_split_predictor.return_value
    assert predictor.data is points2
    predictor.predict.assert_called_with(datetime.datetime(2016,10,5),
            datetime.datetime(2016,10,7), time_samples=5, space_samples=-5)
    assert pred is predictor.predict.return_value
