import open_cp.scripted as scripted
import open_cp.data

import open_cp.sources.chicago
import lzma, datetime, os
import opencrimedata.chicago

import sepp.predictors as predictors
import sepp.grid_nonparam
import sepp.sepp_grid_space
import sepp.sepp_fixed
import sepp.kernels

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
side = "North"

def load_points():
    """Load Chicago data for 2016"""
    filename = os.path.join(datadir, "chicago_all.csv.xz")
    proj = opencrimedata.chicago.projector()
    with lzma.open(filename, "rt") as file:
        times, xcs, ycs = [], [], []
        for row in opencrimedata.chicago.load_only_with_point(file):
            if row.crime_type == "BURGLARY":
                times.append(row.datetime)
                x, y = proj(*row.point)
                xcs.append(x)
                ycs.append(y)
    return open_cp.data.TimedPoints.from_coords(times, xcs, ycs)

def load_geometry():
    """Load the geometry for Chicago; we'll use Southside, as ever..."""
    open_cp.sources.chicago.set_data_directory(datadir)
    return open_cp.sources.chicago.get_side(side)

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:
    
    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))
    
    state.add_prediction(scripted.NaiveProvider, time_range)
    
    for omegai in [1, 2, 5, 10, 20, 100]:
        for sigma in [10, 20, 50, 100]:
            tk = sepp.sepp_fixed.ExpTimeKernel(1/omegai)
            sk = sepp.sepp_fixed.GaussianSpaceKernel(sigma)
            provider = predictors.GridFixedTriggerProvider(state.grid, state.timed_points,
                tk, sk, cutoff=time_range.first)
            state.add_prediction_range(provider, time_range)

    for omegai in [1, 2, 5, 10, 20, 100]:
        for sigma in [10, 20, 50, 100]:
            for bkb in [20, 50, 100, 200]:
                tk = sepp.sepp_fixed.ExpTimeKernel(1/omegai)
                sk = sepp.sepp_fixed.GaussianSpaceKernel(sigma)
                bk = sepp.kernels.FixedBandwidthKernelProvider(bkb)
                provider = predictors.KDEFixedTriggerProvider(state.timed_points,
                    tk, sk, bk, cutoff=time_range.first)
                state.add_prediction_range(provider, time_range)

    state.score(scripted.HitCountEvaluator)
    state.process(scripted.HitCountSave("{}_fixed.csv".format(side)))
