"""`ImageStimulus`, `LogoBVL`, `LogoUCSB`, `SnellenChart`"""
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot

from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import rgba2rgb, rgb2gray
from skimage.measure import moments as img_moments
from skimage.transform import (resize as img_resize, rotate as img_rotate,
                               warp as img_warp, SimilarityTransform)
from skimage.filters import (threshold_mean, threshold_minimum, threshold_otsu,
                             threshold_multiotsu, threshold_local,
                             threshold_isodata)

from .base import Stimulus


class ImageStimulus(Stimulus):
    """ImageStimulus

    .. versionadded:: 0.7

    A stimulus made from an image, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get converted to
    activation values in the range [0, 1].

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.VideoStimulus`

    Parameters
    ----------
    source : str
        Path to image file. Supported image types include JPG, PNG, and TIF;
        and are inferred from the file ending. If the file does not have a
        proper file ending, specify the file type via ``format``.
        Use :py:class:`~pulse2percept.stimuli.VideoStimulus` for GIFs.

    format : str
        An image format string supported by imageio, such as 'JPG', 'PNG', or
        'TIFF'. Use if the file type cannot be inferred from ``source``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will remove pixels with 0 grayscale value.

    """
    __slots__ = ('img_shape',)

    def __init__(self, source, format=None, resize=None, as_gray=False,
                 electrodes=None, metadata=None, compress=False):
        if metadata is None:
            metadata = {}
        if isinstance(source, str):
            # Filename provided:
            img = imread(source, format=format)
            metadata['source'] = source
            metadata['source_shape'] = img.shape
        elif isinstance(source, ImageStimulus):
            img = source.data.reshape(source.img_shape)
            metadata.update(source.metadata)
            if electrodes is None:
                electrodes = source.electrodes
        elif isinstance(source, np.ndarray):
            img = source
        else:
            raise TypeError("Source must be a filename or another "
                            "ImageStimulus, not %s." % type(source))
        if img.ndim < 2 or img.ndim > 4:
            raise ValueError("Images must either have 2 (e.g., grayscale), "
                             "3 (e.g., RGB) or 4 channels (e.g., RGBA), "
                             "not %d." % img.ndim)
        # Convert to grayscale if necessary:
        if as_gray:
            if img.ndim == 4:
                # Convert the transparent background to black:
                img = rgba2rgb(img, background=(0, 0, 0))
            img = rgb2gray(img)
        # Resize if necessary:
        if resize is not None:
            img = img_resize(img, resize)
        # Store the original image shape for resizing and color conversion:
        self.img_shape = img.shape
        # Convert to float array in [0, 1] and call the Stimulus constructor:
        super(ImageStimulus, self).__init__(img_as_float(img).ravel(),
                                            time=None, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)

    def invert(self):
        img = 1.0 - self.data.reshape(self.img_shape)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def rgb2gray(self, electrodes=None):
        img = self.data.reshape(self.img_shape)
        if img.ndim == 4:
            # Convert the transparent background to black:
            img = rgba2rgb(img, background=(0, 0, 0))
        return ImageStimulus(rgb2gray(img), electrodes=electrodes,
                             metadata=self.metadata)

    def threshold(self, thresh, **kwargs):
        """Threshold the image

        Parameters
        ----------
        thresh : str or float
            If a float in [0,1] is provided, pixels whose grayscale value is
            above said threshold will be white, others black.

            A number of additional methods are supported:

            *  'mean': Threshold image based on the mean of grayscale values.
            *  'minimum': Threshold image based on the minimum method, where
                          the histogram of the input image is computed and
                          smoothed until there are only two maxima.
            *  'local': Threshold image based on `local pixel neighborhood
                        <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local>_.
                        Requires ``block_size``: odd number of pixels in the
                        neighborhood.
            *  'otsu': `Otsu's method
                       <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu>_
            *  'multiotsu': `Multi-Otsu method
                            <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu>`_
                            to divide gray levels into multiple classes.
            *  'isodata': `ISODATA method
                          <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata>`_,
                          also known as the Ridler-Calvard method or
                          intermeans.
        """
        if len(self.img_shape) > 2:
            raise ValueError("Thresholding is only supported for grayscale "
                             "(i.e., single-channel) images. Use `rgb2gray` "
                             "first.")
        img = self.data.reshape(self.img_shape)
        if isinstance(thresh, str):
            if thresh.lower() == 'mean':
                img = img > threshold_mean(img)
            elif thresh.lower() == 'minimum':
                img = img > threshold_minimum(img, **kwargs)
            elif thresh.lower() == 'local':
                img = img > threshold_local(img, **kwargs)
            elif thresh.lower() == 'otsu':
                img = img > threshold_otsu(img, **kwargs)
            elif thresh.lower() == 'multiotsu':
                img = np.digitize(img, bins=threshold_multiotsu(img, **kwargs))
            elif thresh.lower() == 'isodata':
                img = img > threshold_isodata(img, **kwargs)
            else:
                raise ValueError("Unknown threshold method '%s'." % thresh)
        elif np.isscalar(thresh):
            img = self.data.reshape(self.img_shape) > thresh
        else:
            raise TypeError("Threshold type must be str or float, not "
                            "%s." % type(thresh))
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def resize(self, shape, electrodes=None):
        img = img_resize(self.data.reshape(self.img_shape), shape)
        return ImageStimulus(img, electrodes=electrodes,
                             metadata=self.metadata)

    def rotate(self, angle, center=None, mode='constant'):
        img = img_rotate(self.data.reshape(self.img_shape), angle, mode=mode,
                         resize=False)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def shift(self, shift_cols, shift_rows):
        """

        Parameters
        ----------
        shift_cols :
            positive, to the right. negative, to the left
        shift_rows :
            positive, upward. negative, downard
        """
        img = self.data.reshape(self.img_shape)
        tf = SimilarityTransform(translation=[shift_cols, shift_rows])
        img = img_warp(img, tf.inverse)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def center(self, loc=None):
        """Centers a phosphene in an image"""
        # Calculate center of mass:
        img = self.data.reshape(self.img_shape)
        m = img_moments(img, order=1)
        # No area found:
        if np.isclose(m[0, 0], 0):
            return img
        # Center location:
        if loc is None:
            loc = np.array(self.img_shape[::-1]) / 2.0 - 0.5
        # Shift the image by -centroid, +image center:
        transl = (loc[0] - m[0, 1] / m[0, 0], loc[1] - m[1, 0] / m[0, 0])
        tf_shift = SimilarityTransform(translation=transl)
        img = img_warp(img, tf_shift.inverse)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def scale(self, scaling_factor):
        # Calculate center of mass:
        img = self.data.reshape(self.img_shape)
        m = img_moments(img, order=1)
        # No area found:
        if np.isclose(m[0, 0], 0):
            return img
        # Shift the phosphene to (0, 0):
        center_mass = np.array([m[0, 1] / m[0, 0], m[1, 0] / m[0, 0]])
        tf_shift = SimilarityTransform(translation=-center_mass)
        # Scale the phosphene:
        tf_scale = SimilarityTransform(scale=scaling_factor)
        # Shift the phosphene back to where it was:
        tf_shift_inv = SimilarityTransform(translation=center_mass)
        # Combine all three transforms:
        tf = tf_shift + tf_scale + tf_shift_inv
        img = img_warp(img, tf.inverse)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def plot(self, kind='pcolor', ax=None, **kwargs):
        """Plot the percept

        Parameters
        ----------
        kind: {'pcolor' | 'hex'}, optional, default: 'pcolor'
            Kind of plot to draw:
            *  'pcolor': using Matplotlib's ``pcolor``. Additional parameters
               (e.g., ``vmin``, ``vmax``) can be passed as keyword arguments.
            *  'hex': using Matplotlib's ``hexbin``. Additional parameters
               (e.g., ``gridsize``) can be passed as keyword arguments.
        ax: matplotlib.axes.Axes; optional, default: None
            A Matplotlib Axes object. If None, a new Axes object will be
            created.

        Returns
        -------
        ax: matplotlib.axes.Axes
            Returns the axes with the plot on it

        """
        frame = self.data.reshape(self.img_shape)
        print(frame.shape)
        if ax is None:
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else:
                figsize = (12, 8)
                # figsize = np.int32(np.array(self.shape[:2][::-1]) / 15)
                # figsize = np.maximum(figsize, 1)
            _, ax = plt.subplots(figsize=figsize)
        else:
            if not isinstance(ax, Subplot):
                raise TypeError("'ax' must be a Matplotlib axis, not "
                                "%s." % type(ax))

        vmin, vmax = frame.min(), frame.max()
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'gray'
        xdva = np.arange(frame.shape[1])
        ydva = np.arange(frame.shape[0])
        X, Y = np.meshgrid(xdva, ydva, indexing='xy')
        if kind == 'pcolor':
            # Create a pseudocolor plot. Make sure to pass additional keyword
            # arguments that have not already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap',
                                                         'vmin', 'vmax'])}
            ax.pcolormesh(X, Y, np.flipud(frame), cmap=cmap, vmin=vmin,
                          vmax=vmax, **other_kwargs)
        elif kind == 'hex':
            # Create a hexbin plot:
            if 'gridsize' in kwargs:
                gridsize = kwargs['gridsize']
            else:
                gridsize = np.min(frame.shape[:2]) // 2
            # X, Y = np.meshgrid(self.xdva, self.ydva, indexing='xy')
            # Make sure to pass additional keyword arguments that have not
            # already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap',
                                                         'gridsize', 'vmin',
                                                         'vmax'])}
            ax.hexbin(X.ravel(), Y.ravel()[::-1], frame.ravel(),
                      cmap=cmap, gridsize=gridsize, vmin=vmin, vmax=vmax,
                      **other_kwargs)
        else:
            raise ValueError("Unknown plot option '%s'. Choose either 'pcolor'"
                             "or 'hex'." % kind)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xdva[0], xdva[-1])
        ax.set_xticks([])
        ax.set_ylim(ydva[0], ydva[-1])
        ax.set_yticks([])
        return ax

    def save(self, fname):
        imsave(fname, self.data.reshape(self.img_shape))


class SnellenChart(ImageStimulus):
    """Snellen chart

    Load a Snellen chart

    Parameters
    ----------
    resize : (height, width) or None
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, white_on_black=True, electrodes=None,
                 metadata=None):
        # Load Snellen chart from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'snellen.jpg')
        # Call ImageStimulus constructor:
        super(SnellenChart, self).__init__(source, format="JPG",
                                           resize=resize,
                                           electrodes=electrodes,
                                           metadata=metadata,
                                           compress=False)


class LogoBVL(ImageStimulus):
    """Bionic Vision Lab (BVL) logo

    Load the 576x720x4 Bionic Vision Lab (BVL) logo.

    Parameters
    ----------
    resize : (height, width) or None
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, electrodes=None, metadata=None,
                 as_gray=False):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'bionic-vision-lab.png')
        # Call ImageStimulus constructor:
        super(LogoBVL, self).__init__(source, format="PNG",
                                      resize=resize,
                                      as_gray=as_gray,
                                      electrodes=electrodes,
                                      metadata=metadata,
                                      compress=False)


class LogoUCSB(ImageStimulus):
    """UCSB logo

    Load a 324x727 white-on-black logo of the University of California, Santa
    Barbara.

    Parameters
    ----------
    resize : (height, width) or None
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, electrodes=None, metadata=None):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'ucsb.png')
        # Call ImageStimulus constructor:
        super(LogoUCSB, self).__init__(source, format="PNG",
                                       resize=resize,
                                       as_gray=True,
                                       electrodes=electrodes,
                                       metadata=metadata,
                                       compress=False)
