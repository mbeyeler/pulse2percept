import numpy as np
import pickle

import skimage.io as skio
import skimage.transform as skit
import skimage.measure as skim
import scipy.optimize as scpo

import pulse2percept as p2p


class InterpSim(p2p.Simulation):
    def __init__(self, implant, name=None, engine='joblib', dojit=True, num_jobs=-1):
        if not isinstance(implant, p2p.implants.ElectrodeArray):
            e_s = "`implant` must be of type p2p.implants.ElectrodeArray"
            raise TypeError(e_s)

        self.name = name
        self.implant = implant
        self.engine = engine
        self.dojit = dojit
        self.num_jobs = num_jobs

        # Optic fiber layer (OFL): After calling `set_optic_fiber_layer`, this
        # variable will contain a `p2p.retina.Grid` object.
        self.ofl = None

        # Ganglion cell layer (GCL): After calling `set_ganglion_cell_layer`,
        # this variable will contain a `p2p.retina.TemporalModel` object.
        self.gcl = None
        
    def set_ganglion_cell_layer(self, loadfile):
        from scipy.interpolate import RegularGridInterpolator
        import pickle
        
        in_list, out_list = pickle.load(open(loadfile, 'rb'))
        in_arr = np.array(in_list)
        amps = np.unique(in_arr[:, 0])
        freqs = np.unique(in_arr[:, 1])
        ecs = np.unique(in_arr[:, 2])
        out_arr = np.array(out_list).reshape((len(amps), len(freqs), len(ecs)))

        self.gcl = RegularGridInterpolator((amps, freqs, ecs), out_arr,
                                           bounds_error=False, fill_value=None)
        
    def pulse2percept(self, amps, freq, layers=['OFL', 'GCL']):
        if 'OFL' in layers:
            ecs, _ = self.ofl.electrode_ecs(self.implant)
        else:
            _, ecs = self.ofl.electrode_ecs(self.implant)
            
        # Sum up current contributions from all electrodes
        ecs = np.sum(ecs[:, :, 1, :] * amps.flatten(), axis=-1)
        
        out_list = np.array([self.gcl([a, freq, 1]) for a in ecs.flatten()])
        out_list[ecs.flatten() < 0] = 0.0
        return out_list.reshape(ecs.shape)


def calc_ssim(target, pred):
    return skim.compare_ssim(target / target.max(),
                             pred / pred.max(),
                             data_range=1.0,
                             gaussian_weights=True, sigma=1.5,
                             use_sample_covariance=False)


def calc_error(target, pred, amps, mode='ssim', lmb=0.0001):
    # reconstruction error
    if mode.lower() == 'ssim':
        reconst = (1.0 - calc_ssim(target, pred)) / 2.0
    elif mode.lower() == 'rmse':
        reconst = skim.compare_nrmse(target / target.max(), pred / pred.max())
    else:
        raise NotImplementedError

    # regularization
    regular = np.linalg.norm(amps.flatten())
    
    return reconst + lmb * regular


def step_model(amps, target, sim, layers, lmb=0.0001):
    pred = sim.pulse2percept(amps, 20, layers=layers)
    err = calc_error(target, pred / pred.max(), amps, mode='ssim', lmb=lmb)
    if np.random.rand() < 0.05:
        print(err)
    return err


letterfile = '../examples/notebooks/letters/G.jpg'
layers = ['GCL']
picklefile = 'xopt-ssim-G-GCL.dat'

implant = p2p.implants.ArgusII()

# The approximated model, interpolated from an input-output function
sim = InterpSim(implant)
sim.set_optic_fiber_layer(sampling=200)
sim.set_ganglion_cell_layer('temporal-model-amps-freqs-ecs.dat')
sim.gcl.grid

img = 255 - skio.imread(letterfile)
img.shape, img.min(), img.max()

img = img[50:150, 70:190]
img_small = skit.resize(img, (6, 10))
img_small.shape, img_small.min(), img_small.max()

target = skit.resize(img, sim.ofl.gridx.shape)
target.shape, target.min(), target.max()

xopt = scpo.fmin(step_model, img_small, args=(target, sim, layers, 0.01),
                 maxfun=500 * img_small.size)

pickle.dump(xopt, open(picklefile, 'wb'))









