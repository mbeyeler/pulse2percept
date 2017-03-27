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

tsample = 0.01 / 1000

r = e2cm.Retina(axon_map='retina_s50_1700by2800.npz',
                sampling=50, ylo=-1700, yhi=1700, xlo=-2800, xhi=2800, axon_lambda=2)

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
print('e_rf', e_rf[0].shape)

loadpath = '/home/mbeyeler/source/pulse2percept/data'

movies = [
#    'boston-train.mp4',
#    'kid-pool.avi',
    'zach-scoot.avi',
    'olly-soccer.avi'
]

framerates = [
#    29,
#    30,
    29,
    119
]

for movie, fps in zip(movies, framerates):
    fstr = '%s-' % movie[:-4]

    print('Processing %s' % movie)
    video = p2p.files.load_video('%s/%s' % (loadpath, movie))
    print('video', video.min(), video.max(), video.dtype, video.shape)

    newvideo = np.zeros((r.gridx.shape[0], r.gridx.shape[1], video.shape[0])).astype(np.float32)
    for i, frame in enumerate(video):
        newframe = sic.rgb2gray(frame).astype(np.float32)
        if newframe.max() > 1.0:
            newframe = newframe / 255.0
        newvideo[..., i] = sit.resize(newframe, r.gridx.shape)
    print('newvideo', newvideo.min(), newvideo.max(), newvideo.dtype, newvideo.shape)
    frames = newvideo
    video = None
    
    frames = np.flipud(frames)
    print('frames', frames.min(), frames.max(), frames.dtype, frames.shape)

    pt = []
    for rf in e_rf:
        rflum = e2cm.retinalmovie2electrodtimeseries( rf, frames, fps=fps)
        # plt.plot(rflum)
        ptrain = e2cm.Movie2Pulsetrain(rflum, tsample=tsample)
        # plt.plot(ptrain.data)
        pt.append(ptrain)

    temporal_model = ec2b.TemporalModel(tsample)

    ecs, _ = r.electrode_ecs(e_all)

    print('Running p2p')
    idx_list, sr_list = ec2b.pulse2percept(temporal_model, ecs, r, pt, n_jobs=5, fps=30)
    pickle.dump((idx_list, sr_list), open(fstr + '-desparatesave.dat', 'wb'))
    print('Done')

    temporal_model = None
    pt = None
    ecs = None

    bm = np.zeros(r.gridx.shape + (sr_list[0].data.shape[-1], ))
    idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    idx_list = None
    bm[idxer] = [sr.data for sr in sr_list] 
    sr_list = None
    
    percept = utils.TimeSeries(tsample, bm)
    percept.resample(20)

    mov = p2p.utils.TimeSeries(tsample, percept.data)

    pickle.dump(mov, open(fstr + '-percept.dat', 'wb'))
