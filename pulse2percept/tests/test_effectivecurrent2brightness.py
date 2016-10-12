import tempfile
import numpy as np
import numpy.testing as npt
import pulse2percept.electrode2currentmap as e2cm
import pulse2percept.effectivecurrent2brightness as ec2b


def test_nanduri_amp_freq():
    # Make sure the Nanduri version of the model can reproduce amp/freq
    # data from Nanduri et al. (2012)
    tsample = 5e-6
    tm = ec2b.TemporalModel(model='Nanduri', tsample=tsample)

    # input amplitude values and expected output
    exp_thresh = 30
    all_amps = np.array([1.25, 1.5, 2, 4, 6]) * exp_thresh
    out_amps = np.array([10, 15, 18, 19, 19])

    # input frequency values and expected output
    all_freqs = np.array([13, 20, 27, 40, 80, 120])
    out_freqs = np.array([7.3, 10, 13, 19, 34, 51])

    ## --- Step 1: Reproduce brightness vs. amplitude ---------------------- ##
    # Run the model on these values and compare model output vs. Nanduri data.
    amp_nanduri = []
    for ampl in all_amps:
        pt = e2cm.Psycho2Pulsetrain(freq=all_freqs[1], dur=0.5,
                                    pulse_dur=0.45 / 1000,
                                    interphase_dur=0, delay=0,
                                    tsample=tsample,
                                    current_amplitude=ampl,
                                    pulsetype='cathodicfirst')
        R4 = tm.model_cascade(pt, dojit=False)
        amp_nanduri.append(R4.data.max())
    amp_nanduri = np.array(amp_nanduri)

    npt.assert_allclose(amp_nanduri, out_amps, rtol=1)

    ## --- Step 2: Reproduce brightness vs. frequency ---------------------- ##
    # Run the model on these values and compare model output vs. Nanduri data.
    freq_nanduri = []
    for freq in all_freqs:
        pt = e2cm.Psycho2Pulsetrain(freq=freq, dur=0.5,
                                    pulse_dur=0.45 / 1000,
                                    interphase_dur=0, delay=0,
                                    tsample=tsample,
                                    current_amplitude=all_amps[0],
                                    pulsetype='cathodicfirst')
        R4 = tm.model_cascade(pt, dojit=False)
        freq_nanduri.append(R4.data.max())
    freq_nanduri = np.array(freq_nanduri)

    npt.assert_allclose(freq_nanduri, out_freqs, rtol=1)


def test_nanduri_vs_krishnan():
    """Test Nanduri vs Krishnan model

    This test miakes sure the Nanduri and Krishnan model flavors give roughly
    the same output. Note: Numerically the models might differ slightly.
    """
    # Choose some reasonable parameter values
    tsample = 1e-5
    tau1 = 4.2e-4
    tau2 = 0.04525
    tau3 = 0.02625
    epsilon = 8.73
    tol = 3

    # Set up both models with the same parameter values
    tm_nanduri = ec2b.TemporalModel(model='Nanduri', tsample=tsample,
                                    tau1=tau1, tau2=tau2, tau3=tau3,
                                    epsilon=epsilon)
    tm_krishnan = ec2b.TemporalModel(model='Krishnan', tsample=tsample,
                                     tau1=tau1, tau2=tau2, tau3=tau3,
                                     epsilon=epsilon)

    # Test a range of reasonable ampl/freq values
    for freq in [5, 10, 20]:
        for ampl in [10, 30, 50]:
            # Define some arbitrary pulse train
            pulse = e2cm.Psycho2Pulsetrain(freq=freq, dur=0.5,
                                           pulse_dur=4.5e-4,
                                           interphase_dur=4.5e-4, delay=0,
                                           tsample=tsample,
                                           current_amplitude=ampl,
                                           pulsetype='cathodicfirst')

            # Apply both models to pulse train
            out_nanduri = tm_nanduri.model_cascade(pulse, dojit=False)
            out_krishnan = tm_krishnan.model_cascade(pulse, dojit=False)

            # Make sure model output doesn't deviate too much
            npt.assert_allclose(np.sum((out_nanduri.data -
                                        out_krishnan.data) ** 2), 0, atol=tol)


def test_brightness_movie():
    retina_file = tempfile.NamedTemporaryFile().name
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, axon_map=retina_file)

    s1 = e2cm.Psycho2Pulsetrain(freq=20, dur=0.5, pulse_dur=.075 / 1000.,
                                interphase_dur=.075 / 1000., delay=0.,
                                tsample=.075 / 1000., current_amplitude=20,
                                pulsetype='cathodicfirst')

    electrode_array = e2cm.ElectrodeArray([1, 1], [0, 1], [0, 1], [0, 1])
    ecs, cs = retina.electrode_ecs(electrode_array)
    temporal_model = ec2b.TemporalModel()
    fps = 30.
    rs = int(1 / (fps * s1.tsample))

    # Smoke testing, feed the same stimulus through both electrodes:
    brightness_movie = ec2b.pulse2percept(temporal_model, ecs, retina,
                                          [s1, s1], rs)

    fps = 30.0
    amplitude_transform = 'linear'
    amp_max = 90
    freq = 20
    pulse_dur = .075 / 1000.
    interphase_dur = .075 / 1000.
    tsample = .005 / 1000.
    pulsetype = 'cathodicfirst'
    stimtype = 'pulsetrain'
    dtype = np.int8
    rflum = np.zeros(100)
    rflum[50] = 1
    m2pt = e2cm.Movie2Pulsetrain(rflum,
                                 fps=fps,
                                 amplitude_transform=amplitude_transform,
                                 amp_max=amp_max,
                                 freq=freq,
                                 pulse_dur=pulse_dur,
                                 interphase_dur=interphase_dur,
                                 tsample=tsample,
                                 pulsetype=pulsetype,
                                 stimtype=stimtype)

    rs = int(1 / (fps * m2pt.tsample))
    # Smoke testing, feed the same stimulus through both electrodes:
    brightness_movie = ec2b.pulse2percept(temporal_model, ecs, retina,
                                          [m2pt, m2pt], rs)

    npt.assert_almost_equal(brightness_movie.tsample,
                            m2pt.tsample * rs,
                            decimal=4)
