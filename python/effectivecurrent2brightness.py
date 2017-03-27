# -*effectivecurrent2brightness -*-
"""effectivecurrent2brightness
This transforms the effective current into brightness for a single point in
space based on the Horsager model as modified by Devyani
Inputs: a vector of effective current over time
Output: a vector of brightness over time
"""
from __future__ import print_function
from scipy.misc import factorial
from scipy.signal import fftconvolve
import scipy.special as ss
import numpy as np
import utils
from utils import TimeSeries


def gamma(n, tau, t):
    """
    returns a gamma function from in [0, t]:

    y = (t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1))

    which is the result of an n stage leaky integrator.
    """

    flag = 0
    if t[0] == 0:
        t = t[1:len(t)]
        flag = 1

    y = ((t/tau)  ** (n-1) *
        np.exp(-t / tau) /
        (tau * factorial(n-1)))

    if flag == 1:
        y = np.concatenate([[0], y])

    return y


class TemporalModel(object):
    def __init__(self, tsample, tau1=.42/1000, tau2=45.25/1000,
                 tau3=26.25/1000, e=8.73, beta=.6, asymptote=14, slope=3,
                 shift=16):
        """
        A model of temporal integration from retina pixels

        Fs : sampling rate

        tau1 = .42/1000  is a parameter for the fast leaky integrator, from
        Alan model, tends to be between .24 - .65

        tau2 = 45.25/1000  integrator for charge accumulation, has values
        between 38-57

        e = scaling factor for the effects of charge accumulation 2-3 for
        threshold or 8-10 for suprathreshold

        tau3 = ??

        parameters for a stationary nonlinearity providing a continuous
        function that nonlinearly rescales the response based on Nanduri et al
        2012, equation 3:

        asymptote = 14
        slope =.3
        shift =47
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.e = e
        self.beta = beta
        self.asymptote = asymptote
        self.slope = slope
        self.shift = shift
        self.tsample = tsample

        t = np.arange(0, 10 * self.tau1, tsample)
        self.gamma1 = gamma(1, self.tau1, t)

        t = np.arange(0, 10 * self.tau2, tsample)
        self.gamma2 = gamma(1, self.tau2, t)

        t = np.arange(0, 10 * self.tau3, tsample)
        self.gamma3 = gamma(3, self.tau3, t)

    def fast_response(self, stimulus, dojit=True):
        """
        Fast response function
        """
        return self.tsample * utils.sparseconv(self.gamma1, stimulus, dojit)

    def charge_accumulation(self, fast_response, stimulus):
        # calculated accumulated charge
        ca = self.tsample * np.cumsum(np.maximum(0, stimulus), axis=-1)
        charge_acc = self.e * self.tsample * fftconvolve(ca, self.gamma2)
        return np.maximum(0, fast_response - charge_acc[:fast_response.size])

    def stationary_nonlinearity(self, fast_response_ca):
        # now we put in the stationary nonlinearity of Devyani's:
        R2norm = fast_response_ca / fast_response_ca.max()
        scale  = ss.expit(fast_response_ca / self.slope - self.shift)
        R3 = R2norm * scale * self.asymptote
        return R3

    def slow_response(self, fast_response_ca_snl):
        # this is cropped as tightly as
        # possible for speed sake
        c = fftconvolve(self.gamma3, fast_response_ca_snl)
        return self.tsample * c

    def calc_pixel(self, ecs_vector, stimuli, dojit):
        ecm = np.zeros(stimuli[0].data.shape[0])  # time vector
        for ii, ecs in enumerate(ecs_vector):
            ecm += ecs * stimuli[ii].data

        fr = self.fast_response(ecm, dojit=dojit)
        ca = self.charge_accumulation(fr, ecm)
        sn = self.stationary_nonlinearity(ca)
        sr = self.slow_response(sn) 
        return TimeSeries(self.tsample, sr)


def pulse2percept(temporal_model, ecs, retina, stimuli,
                  fps=30, dojit=True, n_jobs=-1, tol=1e-10):
    """
    From pulses (stimuli) to percepts (spatio-temporal)

    Parameters
    ----------
    temporal_model : emporalModel class instance.
    ecs : ndarray
    retina : a Retina class instance.
    stimuli : list
    subsample_factor : float/int, optional
    dojit : bool, optional
    """
    rs = int(1 / (fps*stimuli[0].tsample))
    ecs_list = []
    idx_list = []
    for xx in range(retina.gridx.shape[1]):
        for yy in range(retina.gridx.shape[0]):
            if np.all(ecs[yy, xx] < tol):
                pass
            else:
                ecs_list.append(ecs[yy, xx])
                idx_list.append([yy, xx])
    print("selected %d/%d pixels" % (len(ecs_list), np.prod(retina.gridx.shape)))

    sr_list = utils.parfor(temporal_model.calc_pixel, ecs_list, n_jobs=n_jobs,
                           func_args=[stimuli, dojit])

    return idx_list, sr_list
    #bm = np.zeros(retina.gridx.shape + (sr_list[0].data.shape[-1], ))
    #idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    #bm[idxer] = [sr.data for sr in sr_list]
    #return TimeSeries(sr_list[0].tsample, bm)



