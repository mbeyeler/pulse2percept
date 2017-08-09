import numpy as np
import scipy.special as ss
import scipy.spatial as spat
import abc
import six
import os.path
import logging

from pulse2percept import utils


SUPPORTED_LAYERS = ['INL', 'GCL', 'OFL']
SUPPORTED_TEMPORAL_MODELS = ['latest', 'Nanduri2012', 'Horsager2009']


class RetinalGrid(object):

    def __init__(self, x_steps=101, x_range=(-20.0, 20.0), y_steps=101,
                 y_range=(-20.0, 20.0)):
        """Generates a spatial grid representing the retinal coordinate frame

        This function generates the coordinate system for the retina.

        Parameters
        ----------
        x_steps : int, optional, default: 101
            Number of samples to generate along the x axis (horizontal).
        x_range : tuple (xlo, xhi), optional, default: (-20.0, 20.0)
            Lower and upper bounds along the x axis (horizontal) in degrees
            of visual angle.        
        y_steps : int, optional, default: 101
            Number of samples to generate along the y axis (vertical).
        y_range : tuple (ylo, yhi), optional, default: (-20.0, 20.0)
            Lower and upper bounds along the y axis (vertical) in degrees
            of visual angle. The lower hemisphere of the retina (i.e., the
            inferior retina that corresponds to the upper visual field) has
            negative y values.
        """
        xlo, xhi = x_range
        ylo, yhi = y_range
        self.xg, self.yg = np.meshgrid(np.linspace(xlo, xhi, x_steps),
                                       np.linspace(ylo, yhi, y_steps),
                                       indexing='xy')

        # Create KDTree from grid
        pos_xy = np.column_stack((self.xg.ravel(), self.yg.ravel()))
        self.tree = spat.cKDTree(pos_xy)

    def contains(self, x, y=None):
        """Returns True if a point / set of points are within grid limits

        This function returns True if a point is within the range of x, y
        values of the grid.

        Parameters
        ----------
        x : float|vector|array
            If `x` is a float, it is interpreted as the x coordinate of a
            single point. In this case, `y` must be specified.
            If `x` is a Nx1 vector, it is interpreted as the x coordinates of
            N points, for which `y` must be specified.
            If `x` is a Nx2 array, it is interpreted as both x and y
            coordinates of N points.
        y : float|vector, optional, default: None
            The y coordinate of a single point / the y coordinates of a set of
            points.

        Returns
        -------
        idx_valid : bool|vector
            For every specified data point, where it lies within (True) or
            outside (False) grid limits.

        Examples
        --------
        >>> import pulse2percept as p2p
        >>> grid = p2p.retina.RetinalGrid(x_steps=3, x_range=(-1, 1),
        ...                               y_steps=3, y_range=(-1, 1))
        >>> grid.contains(0.95, 0.95)
        True
        >>> grid.contains([1.01, 1.01])
        False
        """
        if y is None:
            xy = np.array(x).reshape((-1, 2))
        else:
            x = np.array([x])
            y = np.array([y])
            xy = np.column_stack((x, y))

        idx_valid = (xy[:, 0] >= self.xg.min()) * (xy[:, 0] <= self.xg.max())
        idx_valid *= (xy[:, 1] >= self.yg.min()) * (xy[:, 1] <= self.yg.max())

        if len(idx_valid) == 1:
            # Return as scalar
            idx_valid = idx_valid[0]
        return idx_valid

    def get_closest_point(self, x, y=None):
        """Returns the closest point on the grid

        This function returns the point on the grid that is closest to the
        specified location.

        Parameters
        ----------
        x : float|vector|array
            If `x` is a float, it is interpreted as the x coordinate of a
            single point. In this case, `y` must be specified.
            If `x` is a Nx1 vector, it is interpreted as the x coordinates of
            N points, for which `y` must be specified.
            If `x` is a Nx2 array, it is interpreted as both x and y
            coordinates of N points.
        y : float|vector, optional, default: None
            The y coordinate of a single point / the y coordinates of a set of
            points.

        Returns
        -------
        x, y : float|vector, float|vector
            The x and y coordinate(s) of the closest point(s) on the grid.

        Examples
        --------
        >>> import pulse2percept as p2p
        >>> grid = p2p.retina.RetinalGrid(x_steps=3, x_range=(-1, 1),
        ...                               y_steps=3, y_range=(-1, 1))
        >>> grid.get_closest_point(0.05, 0.8)
        (0.0, 1.0)
        >>> grid.get_closest_point([0.51, 0.49])
        (1.0, 0.0)
        """
        if y is None:
            xy = np.array(x).reshape((-1, 2))
        else:
            x = np.array([x])
            y = np.array([y])
            xy = np.column_stack((x, y))

        _, idx = self.tree.query(xy)
        if len(idx) == 1:
            # Return a scalar
            idx = idx[0]
        return self.xg.ravel()[idx], self.yg.ravel()[idx]


class Grid(object):
    """Represent the retinal coordinate frame"""

    def __init__(self, x_range=(-1000.0, 1000.0), y_range=(-1000.0, 1000.0),
                 sampling=25, axon_lambda=2.0, datapath='.', save_data=True,
                 engine='joblib', scheduler='threading', n_jobs=-1):
        """Generates a spatial grid representing the retinal coordinate frame

        This function generates the coordinate system for the retina
        and an axon map. As this can take a while, the function will
        first look for an already existing file in the directory `datapath`
        that was automatically created from an earlier call to this function,
        before it attempts to generate new grid from scratch.

        Parameters
        ----------
        x_range : (xlo, xhi), optional, default: xlo=-1000, xhi=1000
           Extent of the retinal coverage (microns) in horizontal dimension.
        y_range : (ylo, yhi), optional, default: ylo=-1000, ylo=1000
           Extent of the retinal coverage (microns) in vertical dimension.
        datapath : str, optional, default: current directory
            Relative path where to look for existing retina files, and where to
            store new files.
        save_data : bool, optional, default: True
            Flag whether to save the data to a new file (True) or not (False).
            The file name is automatically generated from all specified input
            arguments.
        engine : str, optional, default: 'joblib'
            Which computational back end to use:
            - 'serial': Single-core computation
            - 'joblib': Parallelization via joblib (requires `pip install
                        joblib`)
            - 'dask': Parallelization via dask (requires `pip install dask`).
                      Dask backend can be specified via `threading`.
        scheduler : str, optional, default: 'threading'
            Which scheduler to use (irrelevant for 'serial' engine):
            - 'threading': a scheduler backed by a thread pool
            - 'multiprocessing': a scheduler backed by a process pool
        n_jobs : int, optional, default: -1
            Number of cores (threads) to run the model on in parallel.
            Specify -1 to use as many cores as available.
        """
        xlo, xhi = x_range
        ylo, yhi = y_range
        # Include endpoints in meshgrid
        num_x = int((xhi - xlo) / sampling + 1)
        num_y = int((yhi - ylo) / sampling + 1)
        self.gridx, self.gridy = np.meshgrid(np.linspace(xlo, xhi, num_x),
                                             np.linspace(ylo, yhi, num_y),
                                             indexing='xy')

        # Create descriptive filename based on input args
        rot = 0.0
        filename = "retina_s%d_l%.1f_rot%.1f_%dx%d.npz" % (sampling,
                                                           axon_lambda,
                                                           rot / np.pi * 180,
                                                           xhi - xlo,
                                                           yhi - ylo)
        filename = os.path.join(datapath, filename)

        # Bool whether we need to create a new grid
        need_new_grid = True

        # Check if such a file already exists. If so, load parameters and
        # make sure they are the same as specified above. Else, create new.
        if os.path.exists(filename):
            need_new_grid = False
            axon_map = np.load(filename)

            # Verify that the file was created with a consistent grid:
            ax_id = axon_map['axon_id']
            ax_wt = axon_map['axon_weight']
            xlo_am = axon_map['xlo']
            xhi_am = axon_map['xhi']
            ylo_am = axon_map['ylo']
            yhi_am = axon_map['yhi']
            sampling_am = axon_map['sampling']
            axon_lambda_am = axon_map['axon_lambda']

            if 'jan_x' in axon_map and 'jan_y' in axon_map:
                jan_x = axon_map['jan_x']
                jan_y = axon_map['jan_y']
            else:
                jan_x = jan_y = None

            # If any of the dimensions don't match, we need a new retina
            need_new_grid |= xlo != xlo_am
            need_new_grid |= xhi != xhi_am
            need_new_grid |= ylo != ylo_am
            need_new_grid |= yhi != yhi_am
            need_new_grid |= sampling != sampling_am
            need_new_grid |= axon_lambda != axon_lambda_am

            if 'rot' in axon_map:
                # Backwards compatibility for older retina object files that
                # had `rot`
                rot_am = axon_map['rot']
                need_new_grid |= rot != rot_am

        # At this point we know whether we need to generate a new retina:
        if need_new_grid:
            info_str = "File '%s' doesn't exist " % filename
            info_str += "or has outdated parameter values, generating..."
            logging.getLogger(__name__).info(info_str)

            jan_x, jan_y = jansonius(rot=rot)
            dva_x = ret2dva(self.gridx)
            dva_y = ret2dva(self.gridy)
            ax_id, ax_wt = make_axon_map(dva_x, dva_y,
                                         jan_x, jan_y,
                                         axon_lambda=axon_lambda)

            # Save the variables, together with metadata about the grid:
            if save_data:
                np.savez(filename,
                         axon_id=ax_id,
                         axon_weight=ax_wt,
                         jan_x=jan_x,
                         jan_y=jan_y,
                         xlo=[xlo],
                         xhi=[xhi],
                         ylo=[ylo],
                         yhi=[yhi],
                         sampling=[sampling],
                         axon_lambda=[axon_lambda],
                         rot=[rot])

        self.axon_lambda = axon_lambda
        self.rot = rot
        self.sampling = sampling
        self.axon_id = ax_id
        self.axon_weight = ax_wt
        self.jan_x = jan_x
        self.jan_y = jan_y
        self.range_x = self.gridx.max() - self.gridx.min()
        self.range_y = self.gridy.max() - self.gridy.min()

    def current2effectivecurrent(self, cs):
        """

        Converts a current spread map to an 'effective' current spread map, by
        passing the map through a mapping of axon streaks.

        Parameters
        ----------
        cs : array
            The 2D spread map in retinal space

        Returns
        -------
        ecm : array
            The effective current spread, a time-series of the same size as the
            current map, where each pixel is the dot product of the pixel
            values in ecm along the pixels in the list in axon_map, weighted
            by the weights axon map.
        """
        ecs = np.zeros(cs.shape)
        for idx in range(0, len(cs.flat)):
            ecs.flat[idx] = np.dot(cs.flat[self.axon_id[idx]],
                                   self.axon_weight[idx])

        # normalize so the response under the electrode in the ecs map
        # is equal to cs
        scale = np.max(cs) / ecs.flat[np.argmax(cs)]
        ecs = ecs * scale

        # this normalization is based on unit current on the retina producing
        # a max response of 1 based on axonal integration.
        # means that response magnitudes don't change as you increase the
        # length of axonal integration or sampling of the retina
        # Doesn't affect normalization over time, or responses as a function
        # of the anount of current,

        return ecs

    def electrode_ecs(self, implant, alpha=14000, n=1.69):
        """
        Gather current spread and effective current spread for each electrode
        within both the bipolar and the ganglion cell layer

        Parameters
        ----------
        implant : implants.ElectrodeArray
            An implants.ElectrodeArray instance describing the implant.

        alpha : float
            Current spread parameter
        n : float
            Current spread parameter

        Returns
        -------
        ecs : contains n arrays containing the the effective current
            spread within various layers
            for each electrode in the array respectively.

        See also
        --------
        Electrode.current_spread
        """

        cs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                       2, len(implant.electrodes)))
        ecs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                        2, len(implant.electrodes)))

        for i, e in enumerate(implant.electrodes):
            cs[..., 0, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='INL', alpha=alpha, n=n)
            ecs[..., 0, i] = cs[..., 0, i]
            cs[..., 1, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='OFL', alpha=alpha, n=n)
            ecs[:, :, 1, i] = self.current2effectivecurrent(cs[..., 1, i])

        return ecs, cs


@six.add_metaclass(abc.ABCMeta)
class BaseModel():
    """Abstract base class for all models of temporal sensitivity.

    This class provides a standard template for all models of temporal
    sensitivity.
    """

    def set_kwargs(self, warn_inexistent, **kwargs):
        """Overwrite any given keyword arguments

        Parameters
        ----------
        warn_inexistent : bool
            If True, displays a warning message if a keyword is provided that
            is not recognized by the temporal model.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key) and warn_inexistent:
                w_s = "Unknown class attribute '%s'" % key
                logging.getLogger(__name__).warning(w_s)
            setattr(self, key, value)

    def __init__(self, **kwargs):
        self.set_kwargs(True, **kwargs)

    @abc.abstractmethod
    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Abstract base ganglion cell model

        Parameters
        ----------
        in_arr: array-like
            A 2D array specifying the effective current values at a particular
            spatial location (pixel); one value per retinal layer and
            electrode. Dimensions: <#layers x #electrodes>
        pt_list : list
            List of pulse train 'data' containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        use_jit : bool
            If True, applies just-in-time (JIT) compilation to expensive
            computations for additional speed-up (requires Numba).
        """
        pass

    # Static attribute
    tsample = 0.005 / 1000


class Horsager2009(BaseModel):
    """Model of temporal sensitivity (Horsager et al. 2009)

    This class implements the model of temporal sensitivty as described in:
    > A Horsager, SH Greenwald, JD Weiland, MS Humayun, RJ Greenberg,
    > MJ McMahon, GM Boynton, and I Fine (2009). Predicting visual sensitivity
    > in retinal prosthesis patients. Investigative Ophthalmology & Visual
    > Science, 50(4):1483.

    Parameters
    ----------
    tsample : float, optional, default: 0.005 / 1000 seconds
        Sampling time step (seconds).
    tau1 : float, optional, default: 0.42 / 1000 seconds
        Time decay constant for the fast leaky integrater of the ganglion
        cell layer (GCL).
    tau2 : float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the charge accumulation, has values
        between 38 - 57 ms.
    tau3 : float, optional, default: 26.25 / 1000 seconds
        Time decay constant for the slow leaky integrator.
        Default: 26.25 / 1000 s.
    epsilon : float, optional, default: 8.73
        Scaling factor applied to charge accumulation (used to be called
        epsilon).
    beta : float, optional, default: 3.43
        Power nonlinearity applied after half-rectification. The original model
        used two different values, depending on whether an experiment is at
        threshold (`beta`=3.43) or above threshold (`beta`=0.83).
    """

    def __init__(self, **kwargs):
        self.tsample = 0.01 / 1000
        self.tau1 = 0.42 / 1000
        self.tau2 = 45.25 / 1000
        self.tau3 = 26.25 / 1000
        self.epsilon = 2.25
        self.beta = 3.43

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

        _, self.gamma1 = utils.gamma(1, self.tau1, self.tsample)
        _, self.gamma2 = utils.gamma(1, self.tau2, self.tsample)
        _, self.gamma3 = utils.gamma(3, self.tau3, self.tsample)

    def calc_layer_current(self, in_arr, pt_list, layers):
        """Calculates the effective current map of a given layer

        Parameters
        ----------
        in_arr: array-like
            A 2D array specifying the effective current values
            at a particular spatial location (pixel); one value
            per retinal layer and electrode.
            Dimensions: <#layers x #electrodes>
        pt_list : list
            List of pulse train 'data' containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        """
        if 'INL' in layers:
            raise ValueError("The Horsager2009 model does not support an "
                             "inner nuclear layer.")

        if ('GCL' or 'OFL') in layers:
            ecm = np.sum(in_arr[1, :, np.newaxis] * pt_list, axis=0)
        else:
            raise ValueError("Acceptable values for `layers` are: 'GCL', "
                             "'OFL'.")
        return ecm

    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Horsager model cascade

        Parameters
        ----------
        in_arr: array-like
            A 2D array specifying the effective current values
            at a particular spatial location (pixel); one value
            per retinal layer and electrode.
            Dimensions: <#layers x #electrodes>
        pt_list : list
            List of pulse train 'data' containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        use_jit : bool
            If True, applies just-in-time (JIT) compilation to
            expensive computations for additional speed-up
            (requires Numba).
        """
        if 'INL' in layers:
            raise ValueError("The Nanduri2012 model does not support an inner "
                             "nuclear layer.")

        # Although the paper says to use cathodic-first, the code only
        # reproduces if we use what we now call anodic-first. So flip the sign
        # on the stimulus here:
        stim = -self.calc_layer_current(in_arr, pt_list, layers)

        # R1 convolved the entire stimulus (with both pos + neg parts)
        r1 = self.tsample * utils.conv(stim, self.gamma1, mode='full',
                                       method='sparse')[:stim.size]

        # It's possible that charge accumulation was done on the anodic phase.
        # It might not matter too much (timing is slightly different, but the
        # data are not accurate enough to warrant using one over the other).
        # Thus use what makes the most sense: accumulate on cathodic
        ca = self.tsample * np.cumsum(np.maximum(0, -stim))
        ca = self.tsample * utils.conv(ca, self.gamma2, mode='full',
                                       method='fft')[:stim.size]
        r2 = r1 - self.epsilon * ca

        # Then half-rectify and pass through the power-nonlinearity
        r3 = np.maximum(0.0, r2) ** self.beta

        # Then convolve with slow gamma
        r4 = self.tsample * utils.conv(r3, self.gamma3, mode='full',
                                       method='fft')[:stim.size]

        return utils.TimeSeries(self.tsample, r4)


class Nanduri2012(BaseModel):
    """Model of temporal sensitivity (Nanduri et al. 2012)

    This class implements the model of temporal sensitivity as described in:
    > Nanduri, Fine, Horsager, Boynton, Humayun, Greenberg, Weiland (2012).
    > Frequency and Amplitude Modulation Have Different Effects on the Percepts
    > Elicited by Retinal Stimulation. Investigative Ophthalmology & Visual
    > Science January 2012, Vol.53, 205-214. doi:10.1167/iovs.11-8401.

    Parameters
    ----------
    tsample : float, optional, default: 0.005 / 1000 seconds
        Sampling time step (seconds).
    tau1 : float, optional, default: 0.42 / 1000 seconds
        Time decay constant for the fast leaky integrater of the ganglion
        cell layer (GCL).
    tau2 : float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the charge accumulation, has values
        between 38 - 57 ms.
    tau3 : float, optional, default: 26.25 / 1000 seconds
        Time decay constant for the slow leaky integrator.
        Default: 26.25 / 1000 s.
    eps : float, optional, default: 8.73
        Scaling factor applied to charge accumulation (used to be called
        epsilon).
    asymptote : float, optional, default: 14.0
        Asymptote of the logistic function used in the stationary
        nonlinearity stage.
    slope : float, optional, default: 3.0
        Slope of the logistic function in the stationary nonlinearity
        stage.
    shift : float, optional, default: 16.0
        Shift of the logistic function in the stationary nonlinearity
        stage.
    """

    def __init__(self, **kwargs):
        # Set default values of keyword arguments
        self.tau1 = 0.42 / 1000
        self.tau2 = 45.25 / 1000
        self.tau3 = 26.25 / 1000
        self.eps = 8.73
        self.asymptote = 14.0
        self.slope = 3.0
        self.shift = 16.0

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

        # perform one-time setup calculations
        # gamma1 is used for the fast response
        _, self.gamma1 = utils.gamma(1, self.tau1, self.tsample)

        # gamma2 is used to calculate charge accumulation
        _, self.gamma2 = utils.gamma(1, self.tau2, self.tsample)

        # gamma3 is used to calculate the slow response
        _, self.gamma3 = utils.gamma(3, self.tau3, self.tsample)

    def calc_layer_current(self, in_arr, pt_list, layers):
        """Calculates the effective current map of a given layer

        Parameters
        ----------
        in_arr: array-like
            A 2D array specifying the effective current values
            at a particular spatial location (pixel); one value
            per retinal layer and electrode.
            Dimensions: <#layers x #electrodes>
        pt_list : list
            List of pulse train 'data' containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        """
        if 'INL' in layers:
            raise ValueError("The Nanduri2012 model does not support an inner "
                             "nuclear layer.")

        if ('GCL' or 'OFL') in layers:
            ecm = np.sum(in_arr[1, :, np.newaxis] * pt_list, axis=0)
        else:
            raise ValueError("Acceptable values for `layers` are: 'GCL', "
                             "'OFL'.")
        return ecm

    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Nanduri model cascade

        Parameters
        ----------
        in_arr: array-like
            A 2D array specifying the effective current values
            at a particular spatial location (pixel); one value
            per retinal layer and electrode.
            Dimensions: <#layers x #electrodes>
        pt_list : list
            List of pulse train 'data' containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        use_jit : bool
            If True, applies just-in-time (JIT) compilation to
            expensive computations for additional speed-up
            (requires Numba).
        """
        if 'INL' in layers:
            raise ValueError("The Nanduri2012 model does not support an inner "
                             "nuclear layer.")

        # `b1` contains a scaled PulseTrain per layer for this particular
        # pixel: Use as input to model cascade
        b1 = self.calc_layer_current(in_arr, pt_list, layers)

        # Fast response
        b2 = self.tsample * utils.conv(b1, self.gamma1, mode='full',
                                       method='sparse',
                                       use_jit=use_jit)[:b1.size]

        # Charge accumulation
        ca = self.tsample * np.cumsum(np.maximum(0, b1))
        ca = self.tsample * utils.conv(ca, self.gamma2, mode='full',
                                       method='fft')[:b1.size]
        b3 = np.maximum(0, b2 - self.eps * ca)

        # Stationary nonlinearity
        sigmoid = ss.expit((b3.max() - self.shift) / self.slope)
        b4 = b3 * sigmoid * self.asymptote

        # Slow response
        b5 = self.tsample * utils.conv(b4, self.gamma3, mode='full',
                                       method='fft')[:b1.size]

        return utils.TimeSeries(self.tsample, b5)


class TemporalModel(BaseModel):
    """Latest edition of the temporal sensitivity model (experimental)

    This class implements the latest version of the temporal sensitivity
    model (experimental). As such, the model might still change from version
    to version. For more stable implementations, please refer to other,
    published models (see `p2p.retina.SUPPORTED_TEMPORAL_MODELS`).

    Parameters
    ----------
    tsample : float, optional, default: 0.005 / 1000 seconds
        Sampling time step (seconds).
    tau_gcl : float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the fast leaky integrater of the ganglion
        cell layer (GCL).
        This is only important in combination with epiretinal electrode
        arrays.
    tau_inl : float, optional, default: 18.0 / 1000 seconds
        Time decay constant for the fast leaky integrater of the inner
        nuclear layer (INL); i.e., bipolar cell layer.
        This is only important in combination with subretinal electrode
        arrays.
    tau_ca : float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the charge accumulation, has values
        between 38 - 57 ms.
    scale_ca : float, optional, default: 42.1
        Scaling factor applied to charge accumulation (used to be called
        epsilon).
    tau_slow : float, optional, default: 26.25 / 1000 seconds
        Time decay constant for the slow leaky integrator.
    scale_slow : float, optional, default: 1150.0
        Scaling factor applied to the output of the cascade, to make
        output values interpretable brightness values >= 0.
    lweight : float, optional, default: 0.636
        Relative weight applied to responses from bipolar cells (weight
        of ganglion cells is 1).
    aweight : float, optional, default: 0.5
        Relative weight applied to anodic charges (weight of cathodic
        charges is 1).
    slope : float, optional, default: 3.0
        Slope of the logistic function in the stationary nonlinearity
        stage.
    shift : float, optional, default: 15.0
        Shift of the logistic function in the stationary nonlinearity
        stage.
    """

    def __init__(self, **kwargs):
        # Set default values of keyword arguments
        self.tau_gcl = 0.42 / 1000
        self.tau_inl = 18.0 / 1000
        self.tau_ca = 45.25 / 1000
        self.tau_slow = 26.25 / 1000
        self.scale_ca = 42.1
        self.scale_slow = 1150.0
        self.lweight = 0.636
        self.aweight = 0.5
        self.slope = 3.0
        self.shift = 15.0

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

        # perform one-time setup calculations
        _, self.gamma_inl = utils.gamma(1, self.tau_inl, self.tsample)
        _, self.gamma_gcl = utils.gamma(1, self.tau_gcl, self.tsample)

        # gamma_ca is used to calculate charge accumulation
        _, self.gamma_ca = utils.gamma(1, self.tau_ca, self.tsample)

        # gamma_slow is used to calculate the slow response
        _, self.gamma_slow = utils.gamma(3, self.tau_slow, self.tsample)

    def fast_response(self, stim, gamma, method, use_jit=True):
        """Fast response function

        Convolve a stimulus `stim` with a temporal low-pass filter `gamma`.

        Parameters
        ----------
        stim : array
           Temporal signal to process, stim(r,t) in Nanduri et al. (2012).
        use_jit : bool, optional
           If True (default), use numba just-in-time compilation.
        usefft : bool, optional
           If False (default), use sparseconv, else fftconvolve.

        Returns
        -------
        Fast response, b2(r,t) in Nanduri et al. (2012).

        Notes
        -----
        The function utils.sparseconv can be much faster than np.convolve and
        signal.fftconvolve if `stim` is sparse and much longer than the
        convolution kernel.
        The output is not converted to a TimeSeries object for speedup.
        """
        conv = utils.conv(stim, gamma, mode='full', method=method,
                          use_jit=use_jit)

        # Cut off the tail of the convolution to make the output signal
        # match the dimensions of the input signal.
        return self.tsample * conv[:stim.shape[-1]]

    def charge_accumulation(self, ecm):
        """Calculates the charge accumulation

        Charge accumulation is calculated on the effective input current
        `ecm`, as opposed to the output of the fast response stage.

        Parameters
        ----------
        ecm : array-like
            A 2D array specifying the effective current values at a particular
            spatial location (pixel); one value per retinal layer, averaged
            over all electrodes through that pixel.
            Dimensions: <#layers x #time points>
        """
        ca = np.zeros_like(ecm)

        for i in range(ca.shape[0]):
            summed = self.tsample * np.cumsum(np.abs(ecm[i, :]))
            conved = self.tsample * utils.conv(summed, self.gamma_ca,
                                               mode='full', method='fft')
            ca[i, :] = self.scale_ca * conved[:ecm.shape[-1]]
        return ca

    def stationary_nonlinearity(self, stim):
        """Stationary nonlinearity

        Nonlinearly rescale a temporal signal `stim` across space and time,
        based on a sigmoidal function dependent on the maximum value of `stim`.
        This is Box 4 in Nanduri et al. (2012).
        The parameter values of the asymptote, slope, and shift of the logistic
        function are given by self.asymptote, self.slope, and self.shift,
        respectively.

        Parameters
        ----------
        stim : array
           Temporal signal to process, stim(r,t) in Nanduri et al. (2012).

        Returns
        -------
        Rescaled signal, b4(r,t) in Nanduri et al. (2012).

        Notes
        -----
        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # use expit (logistic) function for speedup
        sigmoid = ss.expit((stim.max() - self.shift) / self.slope)
        return stim * sigmoid

    def slow_response(self, stim):
        """Slow response function

        Convolve a stimulus `stim` with a low-pass filter (3-stage gamma)
        with time constant self.tau_slow.
        This is Box 5 in Nanduri et al. (2012).

        Parameters
        ----------
        stim : array
           Temporal signal to process, stim(r,t) in Nanduri et al. (2012)

        Returns
        -------
        Slow response, b5(r,t) in Nanduri et al. (2012).

        Notes
        -----
        This is by far the most computationally involved part of the perceptual
        sensitivity model.
        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # No need to zero-pad: fftconvolve already takes care of optimal
        # kernel/data size
        conv = utils.conv(stim, self.gamma_slow, method='fft', mode='full')

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal.
        return self.scale_slow * self.tsample * conv[:stim.shape[-1]]

    def calc_layer_current(self, ecs_item, pt_list, layers):
        """For a given pixel, calculates the effective current for each retinal
           layer over time

        This function operates at a single-pixel level: It calculates the
        combined current from all electrodes through a spatial location
        over time. This calculation is performed per retinal layer.

        Parameters
        ----------
        ecs_item: array-like
            A 2D array specifying the effective current values at a
            particular spatial location (pixel); one value per retinal
            layer and electrode.
            Dimensions: <#layers x #electrodes>
        pt_list: list
            A list of PulseTrain `data` containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate. Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        """
        not_supported = np.array([l not in SUPPORTED_LAYERS for l in layers],
                                 dtype=bool)
        if any(not_supported):
            raise ValueError("Acceptable values for `layers` is 'OFL', 'GCL', "
                             "'INL'.")

        ecm = np.zeros((ecs_item.shape[0], pt_list[0].shape[-1]))
        if 'INL' in layers:
            ecm[0, :] = np.sum(ecs_item[0, :, np.newaxis] * pt_list, axis=0)
        if ('GCL' or 'OFL') in layers:
            ecm[1, :] = np.sum(ecs_item[1, :, np.newaxis] * pt_list, axis=0)
        return ecm

    def model_cascade(self, ecs_item, pt_list, layers, use_jit):
        """The Temporal Sensitivity model

        This function applies the model of temporal sensitivity to a single
        retinal cell (i.e., a pixel). The model is inspired by Nanduri
        et al. (2012), with some extended functionality.

        Parameters
        ----------
        ecs_item: array-like
            A 2D array specifying the effective current values at a particular
            spatial location (pixel); one value per retinal layer and
            electrode.
            Dimensions: <#layers x #electrodes>
        pt_list: list
            A list of PulseTrain `data` containers.
            Dimensions: <#electrodes x #time points>
        layers : list
            List of retinal layers to simulate. Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        use_jit : bool
            If True, applies just-in-time (JIT) compilation to expensive
            computations for additional speed-up (requires Numba).

        Returns
        -------
        Brightness response over time. In Nanduri et al. (2012), the
        maximum value of this signal was used to represent the perceptual
        brightness of a particular location in space, B(r).
        """
        # For each layer in the model, scale the pulse train data with the
        # effective current:
        ecm = self.calc_layer_current(ecs_item, pt_list, layers)

        # Calculate charge accumulation on the input
        ca = self.charge_accumulation(ecm)

        # Sparse convolution is faster if input is sparse. This is true for
        # the first convolution in the cascade, but not for subsequent ones.
        if 'INL' in layers:
            fr_inl = self.fast_response(ecm[0], self.gamma_inl,
                                        use_jit=use_jit,
                                        method='sparse')

            # Cathodic and anodic parts are treated separately: They have the
            # same charge accumulation, but anodic currents contribute less to
            # the response
            fr_inl_cath = np.maximum(0, -fr_inl)
            fr_inl_anod = self.aweight * np.maximum(0, fr_inl)
            resp_inl = np.maximum(0, fr_inl_cath + fr_inl_anod - ca[0, :])
        else:
            resp_inl = np.zeros_like(ecm[0])

        if ('GCL' or 'OFL') in layers:
            fr_gcl = self.fast_response(ecm[1], self.gamma_gcl,
                                        use_jit=use_jit,
                                        method='sparse')

            # Cathodic and anodic parts are treated separately: They have the
            # same charge accumulation, but anodic currents contribute less to
            # the response
            fr_gcl_cath = np.maximum(0, -fr_gcl)
            fr_gcl_anod = self.aweight * np.maximum(0, fr_gcl)
            resp_gcl = np.maximum(0, fr_gcl_cath + fr_gcl_anod - ca[1, :])
        else:
            resp_gcl = np.zeros_like(ecm[1])

        resp = resp_gcl + self.lweight * resp_inl
        resp = self.stationary_nonlinearity(resp)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)


def ret2dva(r_um):
    """Converts retinal distances (um) to visual angles (deg)

    This function converts an eccentricity measurement on the retinal
    surface (in micrometers), measured from the optic axis, into degrees
    of visual angle.
    Source: Eq. A6 in Watson (2014), J Vis 14(7):15, 1-17
    """
    sign = np.sign(r_um)
    r_mm = 1e-3 * np.abs(r_um)
    r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
    r_deg += 3.027e-4 * r_mm ** 4
    return sign * r_deg


@utils.deprecated(alt_func='p2p.retina.ret2dva', deprecated_version='0.2',
                  removed_version='0.3')
def micron2deg(micron):
    """Transforms a distance from microns to degrees

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    deg = micron / 280.0
    return deg


@utils.deprecated(alt_func='p2p.retina.dva2ret', deprecated_version='0.2',
                  removed_version='0.3')
def deg2micron(deg):
    """Transforms a distance from degrees to microns

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    microns = 280.0 * deg
    return microns


def dva2ret(r_deg):
    """Converts visual angles (deg) into retinal distances (um)

    This function converts a retinal distancefrom the optic axis (um)
    into degrees of visual angle.
    Source: Eq. A5 in Watson (2014), J Vis 14(7):15, 1-17
    """
    sign = np.sign(r_deg)
    r_deg = np.abs(r_deg)
    r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
    r_um = 1e3 * r_mm
    return sign * r_um


def jansonius2009(phi0, n_rho=801, rho_range=(4.0, 45.0),
                  loc_od=(15.0, 2.0), beta_sup=-1.9, beta_inf=0.5):
    """Grows a single axon bundle based on the model by Jansonius et al. (2009)

    This function generates the trajectory of a single nerve fiber bundle
    based on the mathematical model described in:

    > Jansionus et al. (2009). A mathematical description of nerve fiber
    > bundle trajectories and their variability in the human retina. Vis
    > Res 49: 2157-2163.

    Parameters
    ----------
    phi0 : float
        Angular position of the axon at its starting point (polar
        coordinates, degrees). Must be within [-180, 180].
    n_rho : int, optional, default: 801
        Number of sampling points along the radial axis (polar coordinates).
    rho_range : (rho_min, rho_max), optional, default: (4.0, 45.0)
        Lower and upper bounds for the radial position values (polar
        coordinates).
    loc_od : (x_od, y_od), optional, default: (15.0, 2.0)
        Location of the center of the optic disc (x, y) in Cartesian
        coordinates.
    beta_sup : float, optional, default: -1.9
        Scalar value for the superior retina (see Eq. 5, `\beta_s` in the
        paper).
    beta_inf : float, optional, default: 0.5
        Scalar value for the inferior retina (see Eq. 6, `\beta_i` in the
        paper.)

    Returns
    -------
    ax_pos: Nx2 array
        Returns a two-dimensional array of axonal positions, where ax_pos[0, :]
        contains the (x, y) coordinates of the axon segment closest to the
        optic disc, and aubsequent row indices move the axon away from the
        optic disc. Number of rows is at most `n_rho`, but might be smaller if
        the axon crosses the meridian.

    Notes
    -----
    The study did not include axons with phi0 in [-60, 60] deg.
    """
    if np.abs(phi0) > 180.0:
        raise ValueError('phi0 must be within [-180, 180].')
    if n_rho < 1:
        raise ValueError('Number of radial sampling points must be >= 1.')
    if np.any(np.array(rho_range) < 0):
        raise ValueError('rho cannot be negative.')
    if rho_range[0] > rho_range[1]:
        raise ValueError('Lower bound on rho cannot be larger than the '
                         ' upper bound.')
    is_superior = phi0 > 0
    rho = np.linspace(rho_range[0], rho_range[1], n_rho)

    if is_superior:
        # Axon is in superior retina, compute `b` (real number) from Eq. 5:
        b = np.exp(beta_sup + 3.9 * np.tanh(-(phi0 - 121.0) / 14.0))
        # Equation 3, `c` a positive real number:
        c = 1.9 + 1.4 * np.tanh((phi0 - 121.0) / 14.0)
    else:
        # Axon is in inferior retina: compute `b` (real number) from Eq. 6:
        b = -np.exp(beta_inf + 1.5 * np.tanh(-(-phi0 - 90.0) / 25.0))
        # Equation 4, `c` a positive real number:
        c = 1.0 + 0.5 * np.tanh((-phi0 - 90.0) / 25.0)

    # Spiral as a function of `rho`:
    phi = phi0 + b * (rho - rho.min()) ** c

    # Convert to Cartesian coordinates
    xprime = rho * np.cos(np.deg2rad(phi))
    yprime = rho * np.sin(np.deg2rad(phi))

    # Find the array elements where the axon crosses the meridian
    if is_superior:
        # Find elements in inferior retina
        idx = np.where(yprime < 0)[0]
    else:
        # Find elements in superior retina
        idx = np.where(yprime > 0)[0]
    if idx.size:
        # Keep only up to first occurrence
        xprime = xprime[:idx[0]]
        yprime = yprime[:idx[0]]

    # Adjust coordinate system, having fovea=[0, 0] instead of `loc_od`=[0, 0]
    xmodel = xprime + loc_od[0]
    ymodel = yprime
    if loc_od[0] > 0:
        # If x-coordinate of optic disc is positive, use Appendix A
        idx = xprime > -loc_od[0]
    else:
        # Else we need to flip the sign
        idx = xprime < -loc_od[0]
    ymodel[idx] = yprime[idx] + loc_od[1] * (xmodel[idx] / loc_od[0]) ** 2

    # Return as Nx2 array
    return np.vstack((xmodel, ymodel)).T


def grow_axon_bundles(n_axons, phi_range=(-180.0, 180.0), n_rho=801,
                      rho_range=(4.0, 45.0), beta_sup=-1.9, beta_inf=0.5,
                      loc_od=(15.0, 2.0), engine='joblib', n_jobs=-1,
                      scheduler='threading'):
    """Grows axon bundles based on the model by Jansonius et al. (2009)

    This function generates the trajectory of `n_axons` nerve fiber bundles
    based on the mathematical model described in:

    > Jansionus et al. (2009). A mathematical description of nerve fiber
    > bundle trajectories and their variability in the human retina. Vis
    > Res 49: 2157-2163.

    Parameters
    ----------
    n_axons : int
        The number of axons to generate. Their start orientations `phi0` (in
        modified polar coordinates) will be sampled uniformly from `phi_range`.
    phi_range : (lophi, hiphi)
        Range of angular positions of axon fibers at their starting points
        (polar coordinates, degrees) to be sampled uniformly with `n_axons`
        samples. Must be within [-180, 180].
    n_rho : int, optional, default: 801
        Number of sampling points along the radial axis (polar coordinates).
    rho_range : (rho_min, rho_max), optional, default: (4.0, 45.0)
        Lower and upper bounds for the radial position values (polar
        coordinates).
    loc_od : (x_od, y_od), optional, default: (15.0, 2.0)
        Location of the center of the optic disc (x, y) in Cartesian
        coordinates.
    beta_sup : float, optional, default: -1.9
        Scalar value for the superior retina (see Eq. 5, `\beta_s` in the
        paper).
    beta_inf : float, optional, default: 0.5
        Scalar value for the inferior retina (see Eq. 6, `\beta_i` in the
        paper.)
    engine : str, optional, default: 'joblib'
        Which computational back end to use:
        - 'serial': Single-core computation
        - 'joblib': Parallelization via joblib (requires `pip install joblib`)
        - 'dask': Parallelization via dask (requires `pip install dask`). Dask
                  backend can be specified via `threading`.
    scheduler : str, optional, default: 'threading'
        Which scheduler to use (irrelevant for 'serial' engine):
        - 'threading': a scheduler backed by a thread pool
        - 'multiprocessing': a scheduler backed by a process pool
    n_jobs : int, optional, default: -1
        Number of cores (threads) to run the model on in parallel. Specify -1
        to use as many cores as available.

    Returns
    -------
    axons: list of Nx2 arrays
        For every generated axon bundle, returns a two-dimensional array of
        axonal positions, where ax_pos[0, :] contains the (x, y) coordinates of
        the axon segment closest to the optic disc, and aubsequent row indices
        move the axon away from the optic disc. Number of rows is at most
        `n_rho`, but might be smaller if the axon crosses the meridian.
    """
    if n_axons < 1:
        raise ValueError('Number of axons must be >= 1.')
    if np.any(np.abs(phi_range) > 180.0):
        raise ValueError('phi must be within [-180, 180].')
    if phi_range[0] > phi_range[1]:
        raise ValueError('Lower bound on phi cannot be larger than the '
                         'upper bound.')

    phi = np.linspace(phi_range[0], phi_range[1], n_axons)
    func_kwargs = {'n_rho': n_rho, 'rho_range': rho_range,
                   'beta_sup': beta_sup, 'beta_inf': beta_inf,
                   'loc_od': loc_od}
    axons = utils.parfor(jansonius2009, phi, func_kwargs=func_kwargs,
                         engine=engine, scheduler=scheduler, n_jobs=n_jobs)
    return axons


@utils.deprecated(alt_func='p2p.retina.grow_axon_bundles',
                  deprecated_version='0.3', removed_version='0.4')
def jansonius(num_cells=500, num_samples=801, center=np.array([15, 2]),
              rot=0 * np.pi / 180, scale=1, bs=-1.9, bi=.5, r0=4,
              max_samples=45, ang_range=60):
    """Implements the model of retinal axonal pathways by generating a
    matrix of (x,y) positions.

    Assumes that the fovea is at [0, 0]

    Parameters
    ----------
    num_cells : int
        Number of axons (cells).
    num_samples : int
        Number of samples per axon (spatial resolution).
    Center: 2 item array
        The location of the optic disk in dva.

    See:

    Jansonius et al., 2009, A mathematical description of nerve fiber bundle
    trajectories and their variability in the human retina, Vision Research
    """

    # Default parameters:
    #
    # r0 = 4;             %Minumum radius (optic disc size)
    #
    # center = [15,2];    %p.center of optic disc
    #
    # rot = 0*pi/180;    %Angle of rotation (clockwise)
    # scale = 1;             %Scale factor
    #
    # bs = -1.9;          %superior 'b' parameter constant
    # bi = .5;            %inferior 'c' parameter constant
    # ang_range = 60

    # sample space of superior/inferior retina, add them in a 1D array
    # superior is where ang0 > 0
    # this will be the first dimension of the meshgrid
    # inferior should go from -180 to -60? or typo in paper
    # ang0 is \phi_0
    ang0 = np.hstack([np.linspace(ang_range, 180, num_cells / 2),  # superior
                      np.linspace(-180, ang_range, num_cells / 2)])  # inferior

    # from r0=4 to max_samples=45, take num_samples=801 steps
    # this will be the second dimension of the meshgrid
    r = np.linspace(r0, max_samples, num_samples)

    # generate angle and radius matrices from vectors with meshgrid
    ang0mat, rmat = np.meshgrid(ang0, r)

    num_samples = ang0mat.shape[0]
    num_cells = ang0mat.shape[1]

    # index into axons from superior (upper) retina
    sup = ang0mat > 0

    # Set up 'b' parameter:
    b = np.zeros([num_samples, num_cells])

    # Equation 5: upper retina
    b[sup] = np.exp(
        bs + 3.9 * np.tanh(-(ang0mat[sup] - 121) / 14))

    # equation 6: lower retina
    b[~sup] = -np.exp(bi + 1.5 * np.tanh(-(-ang0mat[~sup] - 90) / 25))

    # Set up 'c' parameter:
    c = np.zeros([num_samples, num_cells])

    # equation 3 (fixed typo)
    # Paper says -(angmat-121)/14. Is the - sign the typo?
    c[sup] = 1.9 + 1.4 * np.tanh((ang0mat[sup] - 121) / 14)
    c[~sup] = 1 + .5 * np.tanh((-ang0mat[~sup] - 90) / 25)   # equation 4

    # Here's the main function: spirals as a function of r (equation 1)
    ang = ang0mat + b * (rmat - r0)**c

    # Transform to x-y coordinates
    xprime = rmat * np.cos(ang * np.pi / 180)
    yprime = rmat * np.sin(ang * np.pi / 180)

    # Find where the fibers cross the horizontal meridian
    cross = np.zeros([num_samples, num_cells])
    cross[sup] = yprime[sup] < 0
    cross[~sup] = yprime[~sup] > 0

    # Set Nans to axon paths after crossing horizontal meridian
    id = np.where(np.transpose(cross))

    curr_col = -1
    for i in range(0, len(id[0])):  # loop through axons
        if curr_col != id[0][i]:
            yprime[id[1][i]:, id[0][i]] = np.NaN
            curr_col = id[0][i]

    # Bend the image according to (the inverse) of Appendix A
    xmodel = xprime + center[0]
    ymodel = yprime
    id = xprime > -center[0]
    ymodel[id] = yprime[id] + center[1] * (xmodel[id] / center[0])**2

    #  rotate about the optic disc and scale
    x = scale * (np.cos(rot) * (xmodel - center[0]) + np.sin(rot) *
                 (ymodel - center[1])) + center[0]
    y = scale * (-np.sin(rot) * (xmodel - center[0]) + np.cos(rot) *
                 (ymodel - center[1])) + center[1]

    return x, y


def make_axon_map(xg, yg, jan_x, jan_y, axon_lambda=1, min_weight=0.001):
    axon_id = []
    axon_weight = []
    for idx, _ in enumerate(xg.ravel()):
        cur_xg = xg.ravel()[idx]
        cur_yg = yg.ravel()[idx]
        # find the nearest axon to this pixel
        d = (jan_x - cur_xg) ** 2 + (jan_y - cur_yg) ** 2
        cur_ax_id = np.nanargmin(d)  # index into the current axon

        # `ax_num`: which axon it is
        # `ax_pos_id0`: the point on that axon that is closest to `px`
        [ax_pos_id0, ax_num] = np.unravel_index(cur_ax_id, d.shape)

        dist = 0
        this_id = [idx]
        this_weight = [1.0]
        for ax_pos_id in range(ax_pos_id0 - 1, -1, -1):
            # increment the distance from the starting point
            ax = (jan_x[ax_pos_id + 1, ax_num] - jan_x[ax_pos_id, ax_num])
            ay = (jan_y[ax_pos_id + 1, ax_num] - jan_y[ax_pos_id, ax_num])
            dist += np.sqrt(ax ** 2 + ay ** 2)

            # weight falls off exponentially as distance from axon cell body
            weight = np.exp(-dist / axon_lambda)

            # find the nearest pixel to the current position along the axon
            dist_xg = np.abs(xg[0, :] - jan_x[ax_pos_id, ax_num])
            dist_yg = np.abs(yg[:, 0] - jan_y[ax_pos_id, ax_num])
            nearest_xg_id = dist_xg.argmin()
            nearest_yg_id = dist_yg.argmin()
            nearest_xg = xg[0, nearest_xg_id]
            nearest_yg = yg[nearest_yg_id, 0]

            # if the position along the axon has moved to a new pixel, and the
            # weight isn't too small...
            if weight > min_weight:
                if nearest_xg != cur_xg or nearest_yg != cur_yg:
                    # update the current pixel location
                    cur_xg = nearest_xg
                    cur_yg = nearest_yg

                    # append the list
                    this_weight.append(weight)
                    this_id.append(np.ravel_multi_index((nearest_yg_id,
                                                         nearest_xg_id),
                                                        xg.shape))

        axon_id.append(this_id)
        axon_weight.append(this_weight)
    return axon_id, axon_weight


@utils.deprecated(alt_func='p2p.retina.make_axon_map',
                  deprecated_version='0.3', removed_version='0.4')
def make_axon_map_legacy(xg, yg, jan_x, jan_y, axon_lambda=1, min_weight=.001):
    """Retinal axon map

    Generates a mapping of how each pixel in the retina space is affected
    by stimulation of underlying ganglion cell axons.
    Parameters
    ----------
    xg, yg : array
        meshgrid of pixel locations in units of visual angle sp
    axon_lambda : float
        space constant for how effective stimulation (or 'weight') falls off
        with distance from the pixel back along the axon toward the optic disc
        (default 1 degree)
    min_weight : float
        minimum weight falloff.  default .001

    Returns
    -------
    axon_id : list
        a list, for every pixel, of the index into the pixel in xg,yg space,
        along the underlying axonal pathway.
    axon_weight : list
        a list, for every pixel, of the axon weight into the pixel in xg,yg
        space

    """
    # initialize tuples
    axon_xg = ()
    axon_yg = ()
    axon_dist = ()
    axon_weight = ()
    axon_id = ()

    # loop through pixels as indexed into a single dimension
    for px in range(0, len(xg.flat)):
        # find the nearest axon to this pixel
        d = (jan_x - xg.flat[px])**2 + (jan_y - yg.flat[px])**2
        cur_ax_id = np.nanargmin(d)  # index into the current axon
        [ax_pos_id0, ax_num] = np.unravel_index(cur_ax_id, d.shape)

        dist = 0

        cur_xg = xg.flat[px]
        cur_yg = yg.flat[px]

        # add first values to the list for this pixel
        axon_dist = axon_dist + ([0],)
        axon_weight = axon_weight + ([1],)
        axon_xg = axon_xg + ([cur_xg],)
        axon_yg = axon_yg + ([cur_yg],)
        axon_id = axon_id + ([px],)

        # now loop back along this nearest axon toward the optic disc
        for ax_pos_id in range(ax_pos_id0 - 1, -1, -1):
            # increment the distance from the starting point
            ax = (jan_x[ax_pos_id + 1, ax_num] - jan_x[ax_pos_id, ax_num])**2
            ay = (jan_y[ax_pos_id + 1, ax_num] - jan_y[ax_pos_id, ax_num])**2
            dist += np.sqrt(ax ** 2 + ay ** 2)

            # weight falls off exponentially as distance from axon cell body
            weight = np.exp(-dist / axon_lambda)

            # find the nearest pixel to the current position along the axon
            dist_xg = np.abs(xg[0, :] - jan_x[ax_pos_id, ax_num])
            dist_yg = np.abs(yg[:, 0] - jan_y[ax_pos_id, ax_num])
            nearest_xg_id = dist_xg.argmin()
            nearest_yg_id = dist_yg.argmin()
            nearest_xg = xg[0, nearest_xg_id]
            nearest_yg = yg[nearest_yg_id, 0]

            # if the position along the axon has moved to a new pixel, and the
            # weight isn't too small...
            if weight > min_weight:
                if nearest_xg != cur_xg or nearest_yg != cur_yg:
                    # update the current pixel location
                    cur_xg = nearest_xg
                    cur_yg = nearest_yg

                    # append the list
                    axon_weight[px].append(np.exp(weight))
                    axon_id[px].append(np.ravel_multi_index((nearest_yg_id,
                                                             nearest_xg_id),
                                                            xg.shape))

    return list(axon_id), list(axon_weight)
