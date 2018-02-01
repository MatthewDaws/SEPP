import open_cp.scripted as scripted
import open_cp.data

import open_cp.sources.chicago
import lzma, datetime, os
import opencrimedata.chicago

import sepp.predictors as predictors
import sepp.grid_nonparam
import sepp.sepp_grid_space
import sepp.sepp_fixed

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")

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
    return open_cp.sources.chicago.get_side("North")

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:
    
    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))

    # Takes about 1 hour
    for r0 in [1,5,10,20,50]:
        provider = predictors.GridSpaceExpDecayProvider(state.grid,
                state.timed_points, r0, time_range.first)
        state.add_prediction_range(provider, time_range)

    # Takes about 17 minutes
    for r0 in [1,5,10,20,50]:
        provider = predictors.GridSpaceSimpleProvider(state.grid,
                state.timed_points, r0, time_range.first)
        state.add_prediction_range(provider, time_range)

    # Takes about 1.5 hours
    for r0 in [1,5,10,20,50]:
        for h in [0.1, 0.5, 1]:
            provider = predictors.GridSpaceSimpleHistogramProvider(state.grid,
                state.timed_points, r0, h, time_range.first)
            state.add_prediction_range(provider, time_range)

    # Takes about 4 hours
    for t in [0.5, 1, 1.5]:
        for s in [5, 10, 20]:
            tkp = sepp.sepp_grid_space.FixedBandwidthTimeKernelProvider(t)
            skp = sepp.sepp_grid_space.FixedBandwidthSpaceKernelProvider(s)
            provider = predictors.GridSpaceKDEProvider(state.grid, state.timed_points,
                tkp, skp, time_range.first)
            state.add_prediction_range(provider, time_range)

    state.score(scripted.HitCountEvaluator)
    state.process(scripted.HitCountSave("north2.csv"))

