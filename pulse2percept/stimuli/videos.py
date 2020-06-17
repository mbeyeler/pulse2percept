"""`VideoStimulus`, `BostonTrain`"""
from os.path import dirname, join
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize as img_resize
from skimage import img_as_float
from imageio import get_reader as video_reader

from .base import Stimulus
from ..utils import parfor


class VideoStimulus(Stimulus):
    """VideoStimulus

    .. versionadded:: 0.7

    A stimulus made from a movie file, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get assigned to
    activation values in the range [0, 1].

    The frame rate of the movie is used to infer the time points at which to
    stimulate.

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Parameters
    ----------
    fname : str
        Path to video file. Supported file types include MP4, AVI, MOV, and
        GIF; and are inferred from the file ending. If the file does not have
        a proper file ending, specify the file type via ``format``.

    format : str
        An image format string supported by imageio, such as 'JPG', 'PNG', or
        'TIFF'. Use if the file type cannot be inferred from ``fname``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None, optional, default: None
        A tuple specifying the desired height and the width of the image
        stimulus.

    anti_aliasing : bool, optional, default: False
        Whether to apply a Gaussian filter to smooth the image prior to
        resizing. It is crucial to filter when down-sampling the image to
        avoid aliasing artifacts.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will compress the source data in two ways:
        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

    interp_method : str or int, optional, default: 'linear'
        For SciPy's ``interp1`` method, specifies the kind of interpolation as
        a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next') or as an integer specifying the order of the spline
        interpolator to use.
        Here, 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order; 'previous' and
        'next' simply return the previous or next value of the point.

    extrapolate : bool, optional, default: False
        Whether to extrapolate data points outside the given range.

    """

    def __init__(self, fname, format=None, resize=None, anti_aliasing=False,
                 electrodes=None, metadata=None, compress=False, as_gray=False,
                 interp_method='linear', extrapolate=False):
        # Open the video reader:
        reader = video_reader(fname, format=format)
        # Combine video metadata with user-specified metadata:
        meta = reader.get_meta_data()
        if metadata is not None:
            meta.update(metadata)
        meta['source'] = fname
        # Read the video:
        vid = np.array([frame for frame in reader])
        if vid.ndim != 4:
            raise ValueError("Video must have 4 channels, not %d." % vid.ndim)
        n_time, n_rows, n_cols, n_channels = vid.shape
        # Convert from frames x rows x columns x channels to
        # rows x columns x frames x channels (necessary for skimage):
        vid = vid.transpose((1, 2, 0, 3))
        # Convert to grayscale:
        if as_gray is not None:
            vid = rgb2gray(vid)
            n_channels = 1
        # Downscale:
        if resize is not None:
            vid = img_resize(vid, resize, anti_aliasing=anti_aliasing)
            n_rows, n_cols = resize
        vid = img_as_float(vid)
        # Convert back to space x time:
        if vid.ndim == 4:
            # rows x columns x channels x time
            vid = vid.transpose((0, 1, 3, 2))
        # Infer the time points from the video frame rate:
        time = np.arange(n_time) * meta['fps']
        # Call the Stimulus constructor:
        print('stim constructor')
        super(VideoStimulus, self).__init__(vid.reshape((-1, n_time)),
                                            time=time, electrodes=electrodes,
                                            metadata=meta, compress=compress,
                                            interp_method=None,
                                            extrapolate=False)
        self.rewind()

    def rewind(self):
        """Rewind the iterator"""
        self._next_frame = 0

    def __iter__(self):
        """Iterate over all frames in self.data"""
        self.rewind()
        return self

    def __next__(self):
        """Returns the next frame when iterating over all frames"""
        this_frame = self._next_frame
        #print('next, frame:', this_frame, 'shape:', self.data.shape[-1])
        if this_frame >= self.data.shape[-1]:
            #print('this_frame:', this_frame, 'stop')
            raise StopIteration
        self._next_frame += 1
        return self.data[..., this_frame]

    def play(self, fps=None, ax=None):
        """Animate the percept as HTML with JavaScript

        Parameters
        ----------
        fps : float or None
            If None, uses the percept's time axis. Not supported for
            non-homogeneous time axis.
        ax : matplotlib.axes.Axes; optional, default: None
            A Matplotlib Axes object. If None, a new Axes object will be
            created.

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            A Matplotlib animation object that will play the percept
            frame-by-frame.

        """
        def update(data):
            # print(data.min(), data.max(), data.shape)
            mat.set_data(data)
            return mat

        def data_gen():
            while True:
                yield next(self)

        if self.time is None:
            raise ValueError("Cannot animate a percept with time=None.")

        # There are several options to animate a percept in Jupyter/IPython
        # (see https://stackoverflow.com/a/46878531). Displaying the animation
        # as HTML with JavaScript is compatible with most browsers and even
        # %matplotlib inline (although it can be kind of slow):
        plt.rcParams["animation.html"] = 'jshtml'
        if ax is None:
            ax = plt.gca()
            fig = plt.gcf()
        else:
            fig = ax.figure
        # Rewind the percept and show the first frame:
        self.rewind()
        mat = ax.imshow(next(self), cmap='gray', vmax=self.data.max())
        fig.colorbar(mat)
        plt.close(fig)
        # Determine the frame rate:
        if fps is None:
            interval = np.unique(np.diff(self.time).round(decimals=3))
            if len(interval) > 1:
                raise NotImplementedError
            interval = interval[0]
        else:
            interval = 1000.0 / fps
        # Create the animation:
        ani = FuncAnimation(fig, update, data_gen, interval=interval,
                            save_count=self.data.shape[-1])
        return ani

    def save(self, fname, shape=None, fps=None):
        """Save the percept as an MP4 or GIF

        Parameters
        ----------
        fname : str
            The filename to be created, with the file extension indicating the
            file type. Percepts with time=None can be saved as images (e.g.,
            '.jpg', '.png', '.gif'). Multi-frame percepts can be saved as
            movies (e.g., '.mp4', '.avi', '.mov') or '.gif'.
        shape : (height, width) or None, optional, default: (320,)
            The desired width x height of the resulting image/video.
            Use (h, None) to use a specified height and automatically infer the
            width from the percept's aspect ratio.
            Analogously, use (None, w) to use a specified width.
            If shape is None, width will be set to 320px and height will be
            inferred accordingly.
        fps : float or None
            If None, uses the percept's time axis. Not supported for
            non-homogeneous time axis.

        Notes
        -----
        *  ``shape`` will be adjusted so that width and height are multiples
            of 16 to ensure compatibility with most codecs and players.

        """
        if shape is None:
            # Use 320px width and infer height from aspect ratio:
            shape = (None, 320)
        height, width = shape
        if height is None and width is None:
            raise ValueError('If shape is a tuple, must specify either height '
                             'or width or both.')
        # Infer height or width if necessary:
        if height is None and width is not None:
            height = width / self.data.shape[1] * self.data.shape[0]
        elif height is not None and width is None:
            width = height / self.data.shape[0] * self.data.shape[1]
        # Rescale percept to desired shape:
        data = resize(self.data, (np.int32(height), np.int32(width)))
        data -= data.min()
        if not np.isclose(data.max(), 0):
            data /= data.max() * 255

        if self.time is None:
            # No time component, store as an image:
            imageio.imwrite(fname, data.astype(np.uint8))
        else:
            # With time component, store as a movie:
            if fps is None:
                interval = np.unique(np.diff(self.time))
                if len(interval) > 1:
                    raise NotImplementedError
                fps = 1000.0 / interval[0]
            imageio.mimwrite(fname, data.transpose((2, 0, 1)).astype(np.uint8),
                             fps=fps)
        logging.getLogger(__name__).info('Created %s.' % fname)


class BostonTrain(VideoStimulus):

    def __init__(self, resize=None, anti_aliasing=False, electrodes=None,
                 metadata=None, compress=False, interp_method='linear',
                 extrapolate=False):
        # Load video:
        module_path = dirname(__file__)
        fname = join(module_path, 'data', 'boston-train.mp4')
        super(BostonTrain, self).__init__(fname, resize=resize,
                                          anti_aliasing=anti_aliasing,
                                          electrodes=electrodes,
                                          metadata=metadata,
                                          compress=compress,
                                          interp_method=interp_method,
                                          extrapolate=extrapolate)
