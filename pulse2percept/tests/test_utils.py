import numpy as np
import numpy.testing as npt

from pulse2percept import utils


def test_TimeSeries():
    dur = 0.5
    for tsample in [0.1, 0.01, 0.001]:
        data = np.zeros((3, int(np.round(dur / tsample))))
        data[1] = 1.5
        ts = utils.TimeSeries(tsample, data)

        # Make sure all attributes are created
        npt.assert_equal(hasattr(ts, 'tsample'), True)
        npt.assert_equal(hasattr(ts, 'shape'), True)
        npt.assert_equal(hasattr(ts, 'duration'), True)
        npt.assert_equal(hasattr(ts, 'data'), True)

        # Make sure attribute values are valid
        npt.assert_equal(ts.tsample, tsample)
        npt.assert_equal(ts.shape, data.shape)
        npt.assert_equal(ts.duration, data.shape[-1] * tsample)
        npt.assert_equal(ts[1].data, 1.5)

        # Make sure resampling works as intended
        rsfactor = 10
        orig_shape = ts.shape[-1]
        ts.resample(rsfactor)
        npt.assert_equal(ts.data.shape[-1], np.ceil(orig_shape / rsfactor))


def test_Parameters():
    my_params = utils.Parameters(foo='bar', list=[1, 2, 3])
    assert my_params.foo == 'bar'
    assert my_params.list == [1, 2, 3]
    assert str(my_params) == 'foo : bar\nlist : [1, 2, 3]'
    my_params.tuple = (1, 2, 3)
    assert my_params.tuple == (1, 2, 3)


def test_sparseconv():
    # time vector for stimulus (long)
    maxT = .5  # seconds
    nt = 100000
    t = np.linspace(0, maxT, nt)

    # stimulus (10 Hz anondic and cathodic pulse train)
    stim = np.zeros(nt)
    stim[0:nt:10000] = 1
    stim[100:nt:1000] = -1

    # time vector for impulse response (shorter)
    tt = t[t < .1]

    # impulse reponse (kernel)
    G = np.exp(-tt / .005)

    # make sure sparseconv returns the same result as np.convolve
    # for all modes
    modes = ["full", "valid", "same"]
    for mode in modes:
        # np.convolve
        conv = np.convolve(stim, G, mode=mode)

        # utils.sparseconv
        sparse_conv = utils.sparseconv(G, stim, mode=mode, dojit=False)

        npt.assert_equal(conv.shape, sparse_conv.shape)
        npt.assert_almost_equal(conv, sparse_conv)


# We define a function of the right form:
def power_it(num, n=2):
    return num ** n


def test_parfor():
    my_array = np.arange(100).reshape(10, 10)
    i, j = np.random.randint(0, 9, 2)
    my_list = list(my_array.ravel())
    npt.assert_equal(utils.parfor(power_it, my_list,
                                  out_shape=my_array.shape)[i, j],
                     power_it(my_array[i, j]))

    # If it's not reshaped, the first item should be the item 0, 0:
    npt.assert_equal(utils.parfor(power_it, my_list)[0],
                     power_it(my_array[0, 0]))


def test_traverse_randomly():
    # Generate a list of integers
    sequence = np.arange(100).tolist()

    # Iterate the sequence in random order
    shuffled_idx = []
    shuffled_val = []
    for idx, value in enumerate(utils.traverse_randomly(sequence)):
        shuffled_idx.append(idx)
        shuffled_val.append(value)

    # Make sure every element was visited once
    npt.assert_equal(shuffled_idx, np.arange(len(sequence)).tolist())
    npt.assert_equal(shuffled_val.sort(), sequence.sort())
