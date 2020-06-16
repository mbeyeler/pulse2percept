import os
import numpy as np
import numpy.testing as npt
import pytest

from skimage.io import imsave

from pulse2percept.stimuli import ImageStimulus, LogoBVL, LogoUCSB


def test_ImageStimulus():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    ndarray = np.random.rand(*shape)
    imsave(fname, (255 * ndarray).astype(np.uint8))

    # Make sure ImageStimulus loaded is identical to dummy image:
    stim = ImageStimulus(fname)
    npt.assert_equal(stim.shape, (np.prod(shape), 1))
    npt.assert_almost_equal(stim.data, ndarray.reshape((-1, 1)), decimal=2)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_shape'], shape)
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.electrodes, np.arange(np.prod(shape)))
    os.remove(fname)

    # Resize the dummy image:
    ndarray = np.ones(shape)
    imsave(fname, (255 * ndarray).astype(np.uint8))
    resize = (12, 18)
    stim = ImageStimulus(fname, resize=resize)
    npt.assert_equal(stim.shape, (np.prod(resize), 1))
    npt.assert_almost_equal(stim.data, np.ones((np.prod(resize), 1)),
                            decimal=2)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_shape'], shape)
    os.remove(fname)


def test_LogoBVL():
    logo = LogoBVL()
    npt.assert_equal(logo.img_shape, (576, 720, 4))
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)


def test_LogoUCSB():
    logo = LogoUCSB()
    npt.assert_equal(logo.img_shape, (324, 727))
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)
