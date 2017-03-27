import sys
sys.path.append('..')

import numpy as np
import electrode2currentmap as e2cm
import effectivecurrent2brightness as ec2b
from scipy import interpolate
from utils import TimeSeries
import utils
import pickle

import pulse2percept as p2p
import skimage.io as sio
import skimage.color as sic
import skimage.transform as sit

r = e2cm.Retina(axon_map='retina_s25_1700by2800.npz',
                sampling=25, ylo=-1700, yhi=1700, xlo=-2800, xhi=2800, axon_lambda=2)

xlist = []
ylist = []
rlist = []
e_spacing = 525

for x in np.arange(-2362, 2364, e_spacing):
    for y in np.arange(-1312, 1314, e_spacing):
        xlist.append(x)
        ylist.append(y)
        rlist.append(100)

e_all = e2cm.ElectrodeArray(rlist, xlist, ylist)

e_rf = []
for e in e_all.electrodes:
    e_rf.append(e2cm.receptive_field(e, r.gridx, r.gridy, e_spacing))

loadpath = '~/source/pulse2percept/examples/notebooks/letters'

letters = [
    'T',
    'O',
    'A'
]

i = 0
j = 0

for letter in letters:
    for invert in [True, False]:
        img = sio.imread('%s/%s.jpg' % (loadpath, letter))
        img = img[25:175, 25:225]
        img.shape

        frames = sic.gray2rgb(img)
        frames = sit.resize(frames, r.gridx.shape)
        if invert:
            frames = 1.0 - frames
        frames = np.flipud(frames)
        frames.min(), frames.max(), frames.dtype, frames.shape

        if invert:
            fstr = '%s-invert-' % letter
        else:
            fstr = '%s-' % letter

        pt = []
        for rf in e_rf:
            rflum = e2cm.retinalmovie2electrodtimeseries(
                rf, frames[i:i + r.gridx.shape[0], j:j + r.gridx.shape[1]], fps=6)
            # plt.plot(rflum)
            ptrain = e2cm.Movie2Pulsetrain(rflum)
            # plt.plot(ptrain.data)
            pt.append(ptrain)

        temporal_model = ec2b.TemporalModel()

        ecs, cs = r.electrode_ecs(e_all)

        brightness_movie = ec2b.pulse2percept(temporal_model, ecs, r, pt)

        mov = p2p.utils.TimeSeries(
            brightness_movie.tsample, brightness_movie.data)

        pickle.dump(mov, open(fstr + '-percept.dat', 'wb'))

