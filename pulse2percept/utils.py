"""
Utility functions for pulse2percept
"""
import numpy as np
import multiprocessing
import random
import copy
import logging

from scipy import misc as spm
from scipy import interpolate as spi
from scipy import signal as sps


# Rather than trying to import these all over, try once and then remember
# by setting a flag.
try:
    import joblib
    joblib.Parallel
    has_joblib = True
except (ImportError, AttributeError):
    has_joblib = False

try:
    import dask
    import dask.multiprocessing
    dask.delayed
    has_dask = True
except (ImportError, AttributeError):
    has_dask = False

try:
    from numba import jit
    has_jit = True
except ImportError:
    has_jit = False


def deprecated(arg):
    """Decorator used to mark functions as deprecated

    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.

    Adapted from: http://www.artima.com/weblogs/viewpost.jsp?thread=240845

    Parameters
    ----------
    alt_func : str, optional
        The function to use instead of the deprecated one.
        Be sure to list a string, not a function object.
        Default: None.
    """
    def wrap(func):
        """The function performing the decoration

        This function is perfoming the actual decoration process. It can only
        have a single argument, which is the function object.

        Parameters
        ----------
        func : function object
            The function that is deprecated.
        """
        e_s = "Call to deprecated function %s." % func.__name__
        if alt_func:
            e_s += " Use %s instead." % alt_func
        logging.getLogger(__name__).warn(e_s)

        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped_func

    alt_func = None
    if callable(arg):
        # Direct decoration
        return wrap(arg)
    else:
        # Alternative function given
        alt_func = arg
        return wrap


class Parameters(object):
    """Container to wrap a MATLAB array in a Python dict

    This function is deprecated as of v0.2 and will be completely removed
    in v0.3.

    .. deprecated:: 0.2
    """

    @deprecated
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v

    def __repr__(self):
        my_list = []
        for k, v in self.__dict__.items():
            my_list.append("%s : %s" % (k, v))
        my_list.sort()
        my_str = "\n".join(my_list)
        return my_str

    def __setattr(self, name, values):
        self.__dict__[name] = values


class TimeSeries(object):

    def __init__(self, tsample, data):
        """Container for time series data

        Provides a container for time series data. Every time series has a
        sampling step `tsample`, and some `data` sampled at that rate.

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds).
        data : array_like
            Time series data sampled at every `tsample` seconds.
        """
        self.data = data
        self.tsample = tsample
        self.duration = self.data.shape[-1] * tsample
        self.shape = data.shape

    def __getitem__(self, y):
        return TimeSeries(self.tsample, self.data[y])

    def append(self, other):
        """Appends the data of another TimeSeries object (in time)

        This function concatenates the data of another TimeSeries object to
        the current object, along the last dimension (time). To make this work,
        all but the last dimension of the two objects must be the same.

        If the two objects have different time sampling steps, the other object
        is resampled to fit the current `tsample`.

        Parameters
        ----------
        other : p2p.utils.TimeSeries
            A TimeSeries object whose content should be appended.

        Examples
        --------
        >>> from pulse2percept import utils
        >>> pt = utils.TimeSeries(1.0, np.zeros((2, 2, 10)))
        >>> num_frames = pt.shape[-1]
        >>> pt.append(pt)
        >>> pt.shape[-1] == 2 * num_frames
        True
        """
        # Make sure type is correct
        if not isinstance(other, TimeSeries):
            raise TypeError("Other object must be of type "
                            "p2p.utils.TimeSeries.")

        # Make sure size is correct for all but the last dimension (number
        # of frames)
        if self.shape[:-1] != other.shape[:-1]:
            raise ValueError("Shape mismatch: ", self.shape[:-1], " vs. ",
                             other.shape[:-1])

        # Then resample the other to current `tsample`
        resampled = other.resample(self.tsample)

        # Then concatenate the two
        self.data = np.concatenate((self.data, resampled.data), axis=-1)
        self.duration = self.data.shape[-1] * self.tsample
        self.shape = self.data.shape

    def max(self):
        """Returns the time and value of the largest data point

        This function returns the the largest value in the TimeSeries data,
        as well as the time at which it occurred.

        Returns
        -------
        t : float
            time (s) at which max occurred
        val : float
            max value
        """
        # Find index and value of largest element
        idx = self.data.argmax()
        val = self.data.max()

        # Find frame that contains the brightest data point using `unravel`,
        # which maps the flat index `idx_px` onto the high-dimensional
        # indices (x,y,z).
        # What we want is index `z` (i.e., the frame index), given by the last
        # dimension in the return argument.
        idx_frame = np.unravel_index(idx, self.data.shape)[-1]

        # Convert index to time
        t = idx_frame * self.tsample

        return t, val

    def max_frame(self):
        """Returns the time frame that contains the largest data point

        This function returns the time frame in the TimeSeries data that
        contains the largest data point, as well as the time at which
        it occurred.

        Returns
        -------
        t : float
            time (s) at which max occurred
        frame : TimeSeries
            A TimeSeries object.
        """
        # Find index and value of largest element
        idx = self.data.argmax()

        # Find frame that contains the brightest data point using `unravel`,
        # which maps the flat index `idx_px` onto the high-dimensional
        # indices (x,y,z).
        # What we want is index `z` (i.e., the frame index), given by the last
        # dimension in the return argument.
        idx_frame = np.unravel_index(idx, self.data.shape)[-1]
        t = idx_frame * self.tsample
        frame = self.data[..., idx_frame]

        return t, TimeSeries(self.tsample, copy.deepcopy(frame))

    def resample(self, tsample_new):
        """Returns data sampled according to new time step

        This function returns a TimeSeries object whose data points were
        resampled according to a new time step `tsample_new`. New values
        are found using linear interpolation.

        Parameters
        ----------
        tsample_new : float
            New sampling time step (s)

        Returns
        -------
        TimeSeries object
        """
        if tsample_new is None or tsample_new == self.tsample:
            return TimeSeries(self.tsample, self.data)

        # Try to avoid rounding errors in arr size by making sure `t_old` is
        # too long at first, then cutting it to the right size
        y_old = self.data
        t_old = np.arange(0, self.duration + self.tsample, self.tsample)
        t_old = t_old[:y_old.shape[-1]]
        f = spi.interp1d(t_old, y_old, axis=-1, fill_value='extrapolate')

        t_new = np.arange(0, self.duration, tsample_new)
        y_new = f(t_new)

        return TimeSeries(tsample_new, y_new)


def sparseconv(data, kernel, mode):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.
    output is of length len(kernel) + len(data) -1 (same as the default for
    numpy.convolve).

    Can run faster than numpy.convolve if:
    (1) `data` is much longer than `kernel`
    (2) `data` is sparse (has lots of zeros)
    """
    kernel_len = kernel.shape[-1]
    data_len = data.shape[-1]
    out = np.zeros(data_len + kernel_len - 1)

    pos = np.where(data != 0)[0]
    # Add shifted and scaled copies of `kernel` only where `data` is nonzero
    for p in pos:
        out[p:p + kernel_len] = out[p:p + kernel_len] + kernel * data[p]

    if mode.lower() == 'full':
        return out
    elif mode.lower() == 'valid':
        return center_vector(out, data_len - kernel_len + 1)
    elif mode.lower() == 'same':
        return center_vector(out, data_len)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")


if has_jit:
    sparseconv = jit(sparseconv)


def conv(data, kernel, mode='full', method='fft', dojit=True):
    """Convoles data with a kernel using either FFT or sparseconv

    This function convolves data with a kernel, relying either on the
    fast Fourier transform (FFT) or a sparse convolution function.

    Parameters
    ----------
    data : array_like
        First input, typically the data array
    kernel : array_like
        Second input, typically the kernel
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
        The output is the full discrete linear convolution of the inputs.
        (Default)
        ``valid``
        The output consists only of those elements that do not rely on
        zero-padding.
        ``same``
        The output is the same size as `data`, centered with respect to the
        'full' output.
    method : str {'fft', 'sparse'}, optional
        A string indicating the convolution method:
        ``fft``
        Use the fast Fourier transform (FFT). (Default)
        ``sparse``
        Use the sparse convolution.
    dojit : bool, optional
        A flag indicating whether to use numba's just-in-time compilation
        option (only relevant for `method`=='sparse').
    """
    if method.lower() == 'fft':
        # Use FFT: faster on non-sparse data
        conved = sps.fftconvolve(data, kernel, mode)
    elif method.lower() == 'sparse':
        # Use sparseconv: faster on sparse data
        if dojit and not has_jit:
            e_s = ("You do not have numba ",
                   "please run sparseconv with dojit=False")
            raise ImportError(e_s)

        conved = sparseconv(data, kernel, mode)
    else:
        raise ValueError("Acceptable methods are: 'fft', 'sparse'.")
    return conved


def center_vector(vec, newlen):
    """
    Returns the center `newlen` portion of a vector.

    Adapted from scipy.signal.signaltools._centered:
    github.com/scipy/scipy/blob/v0.18.0/scipy/signal/signaltools.py#L236-L243

    """
    currlen = vec.shape[-1]
    startind = (currlen - newlen) // 2
    endind = startind + newlen
    return vec[startind:endind]


def parfor(func, in_list, out_shape=None, n_jobs=-1, engine='joblib',
           backend='threading', func_args=[], func_kwargs={}):
    """
    Parallel for loop for numpy arrays

    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        func(arr, idx, *args, *kwargs) where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of `func` needs to be one
        item (e.g. float, int) per input item.

    in_list : list
       All legitimate inputs to the function to operate over.

    n_jobs : integer, optional
        The number of jobs to perform in parallel. -1 to use all cpus
        Default: 1

    engine : str
        {"dask", "joblib", "serial"}
        The last one is useful for debugging -- runs the code without any
        parallelization.

    backend : str
        What dask backend to use. Irrelevant for other engines.

    func_args : list, optional
        Positional arguments to `func`

    func_kwargs : list, optional
        Keyword arguments to `func`

    Returns
    -------
    ndarray of identical shape to `arr`

    Notes
    -----
    Imported from pyAFQ (blob e20eaa0 from June 3, 2016):
    https://github.com/arokem/pyAFQ/blob/master/AFQ/utils/parallel.py

    Examples
    --------
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        n_jobs = n_jobs - 1

    if engine.lower() == 'joblib':
        if not has_joblib:
            err = "You do not have `joblib` installed. Consider setting"
            err += "`engine` to 'serial' or 'dask'."
            raise ImportError(err)

        p = joblib.Parallel(n_jobs=n_jobs, backend=backend)
        d = joblib.delayed(func)
        d_l = []
        for in_element in in_list:
            d_l.append(d(in_element, *func_args, **func_kwargs))
        results = p(d_l)
    elif engine.lower() == 'dask':
        if not has_dask:
            err = "You do not have `dask` installed. Consider setting"
            err += "`engine` to 'serial' or 'joblib'."
            raise ImportError(err)

        def partial(func, *args, **keywords):
            def newfunc(in_arg):
                return func(in_arg, *args, **keywords)
            return newfunc
        p = partial(func, *func_args, **func_kwargs)
        d = [dask.delayed(p)(i) for i in in_list]
        if backend == 'multiprocessing':
            results = dask.compute(*d, get=dask.multiprocessing_get,
                                   workers=n_jobs)
        elif backend == 'threading':
            results = dask.compute(*d, get=dask.threaded.get, workers=n_jobs)
    elif engine.lower() == 'serial':
        results = []
        for in_element in in_list:
            results.append(func(in_element, *func_args, **func_kwargs))

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results


@deprecated
def mov2npy(movie_file, out_file):
    """Converts a movie file to a NumPy array

    This function is deprecated as of v0.2 and will be removed completely
    in v0.3.

    Parameters
    ----------
    movie_file : array
        The movie file to be read
    out_file: str
        Name of .npy file to be created.

    .. deprecated:: 0.2
    """
    # Don't import cv at module level. Instead we'll use this on python 2
    # sometimes...
    try:
        import cv
    except ImportError:
        e_s = "You do not have opencv installed. "
        e_s += "You probably want to run this in Python 2"
        raise ImportError(e_s)

    capture = cv.CaptureFromFile(movie_file)
    frames = []
    img = cv.QueryFrame(capture)
    while img is not None:
        tmp = cv.CreateImage(cv.GetSize(img), 8, 3)
        cv.CvtColor(img, tmp, cv.CV_BGR2RGB)
        frames.append(np.asarray(cv.GetMat(tmp)))
        img = cv.QueryFrame(capture)
    frames = np.fliplr(np.rot90(np.mean(frames, -1).T, -1))
    np.save(out_file, frames)


def gamma(n, tau, tsample, tol=0.01):
    """Returns the impulse response of `n` cascaded leaky integrators

    This function calculates the impulse response of `n` cascaded
    leaky integrators with constant of proportionality 1/`tau`:
    y = (t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1))

    Parameters
    ----------
    n : int
        Number of cascaded leaky integrators
    tau : float
        Decay constant of leaky integration (seconds).
        Equivalent to the inverse of the constant of proportionality.
    tsample : float
        Sampling time step (seconds).
    tol : float
        Cut the kernel to size by ignoring function values smaller
        than a fraction `tol` of the peak value.
    """
    n = int(n)

    # Allocate a time vector that is long enough for sure.
    # Trim vector later on.
    t = np.arange(0, 5 * n * tau, tsample)

    # Calculate gamma
    y = (t / tau) ** (n - 1) * np.exp(-t / tau)
    y /= (tau * spm.factorial(n - 1))

    # Normalize to unit area
    y /= np.trapz(np.abs(y), dx=tsample)

    # Cut off tail where values are smaller than `tol`.
    # Make sure to start search on the right-hand side of the peak.
    peak = y.argmax()
    small_vals = np.where(y[peak:] < tol * y.max())[0]
    t = t[:small_vals[0] + peak]
    y = y[:small_vals[0] + peak]

    return t, y


def traverse_randomly(seq):
    """Traverses a list in random order

    Parameters
    ----------
    seq : list
        An iterable list of elements.
    Returns
    -------
    A list iterator.

    Examples
    --------
    Shuffle a list:

    >>> list_ordered = [0, 1, 2, 3]
    >>> list_shuffled = [l for l in traverse_randomly(list_ordered)]

    Traverse a list in random order:

    >>> list_ordered = [0, 1, 2, 3]
    >>> for idx, val in traverse_randomly(enumerate(list_ordered)):
    ...     pass

    Notes
    -----
    From: http://stackoverflow.com/a/9253366
    Note that even for rather small `len(x)`, the total number of permutations
    of `seq` can quickly grow larger than the period of most random number
    generators. For example, a sequence of length 2080 is the largest that can
    fit within the period of the Mersenne Twister random number generator.
    """
    # Make sure we are dealing with a list
    shuffled = list(seq)

    # Import `random` at module level (for performance)
    random.shuffle(shuffled)

    return iter(shuffled)
