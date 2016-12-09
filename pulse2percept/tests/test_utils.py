from pulse2percept import utils
import numpy as np
import numpy.testing as npt
from nose import SkipTest


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


def test_CuFFTConvolve():
    try:
        from scipy.signal import fftconvolve
        import pycuda.autoinit
        import skcuda.fft
    except ImportError:
        raise SkipTest("Missing either scipy's fftconvolve, PyCUDA, or "
                       "scikit-cuda.")

    ## ---- Step 1: Constructor and setup ---------------------------------- ##
    # Only supported mode is 'full' so far
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(10, 10, 'same')
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(10, 10, 'valid')
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(10, 10, 'whatever')

    # Array shapes must be valid
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(0, 2)
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(3, 0)
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(-1, 2)
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(2, -1)

    # Array shapes cannot change from constructor to method call
    cu = utils.CuFFTConvolve(10, 10)
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(np.ones(9), np.ones(10))
    with pytest.raises(ValueError):
        utils.CuFFTConvolve(np.ones(10), np.ones(11))

    # Output size varies with mode
    for in1size in [2, 12, 1000]:
        for in2size in [5, 20, 500]:
            for mode in ['full']:
                cu = utils.CuFFTConvolve(in1size, in2size, mode)

                if mode == 'full':
                    npt.assert_equal(cu.out_size, in1size + in2size - 1)

    ## ---- Step 2: Test cufftconvolve with different (realistic) input ---- ##
    # Precision will vary depending on array length and GPU architecture (some
    # GPUs have more rounding errors)
    in1 = []      # first input signal
    in2 = []      # second input signal
    decimal = []  # precision (almost equal to how many decimals)

    # Some easy and short signal, same size, high precision
    in1.append(np.sin(np.linspace(0, 10, 100)))
    in2.append(np.cos(np.linspace(0, 3, 100)))
    decimal.append(5)

    # Small second kernel, high precision
    in1.append(np.sin(np.linspace(0, 10, 100)))
    in2.append(np.cos(np.linspace(0, 3, 19)))
    decimal.append(5)

    # Longer kernel, lower precision
    in1.append(np.sin(np.linspace(0, 10, 10000)))
    in2.append(np.cos(np.linspace(0, 5, 1000)))
    decimal.append(3)

    # Let's use some signals with lengths that are relevant to pulse2percept:
    dur_pt = 0.5                   # typical pulse train duration
    tsample = 5e-6                 # typical sampling step
    dur_gamma3 = 8 * 26.25 / 1000  # typical gamma3 duration
    in1.append(np.random.rand(int(dur_pt / tsample)))
    in2.append(np.random.rand(int(dur_gamma3 / tsample)))
    decimal.append(2)

    for x1, x2, dec in zip(in1, in2, decimal):
        # scipy's version:
        y_cpu = fftconvolve(x1, x2)

        # our cuda version:
        cu = utils.CuFFTConvolve(x1.size, x2.size, mode='full')
        y_gpu = cu.cufftconvolve(x1, x2)

        # Make sure they're almost equal
        npt.assert_almost_equal(y_cpu, y_gpu, decimal=dec)
