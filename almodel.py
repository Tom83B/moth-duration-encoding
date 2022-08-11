from brian2 import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import pandas as pd
from loaders import DurationExperiment
import pickle
from pathos.multiprocessing import ProcessingPool as Pool

from utils import get_filenames, get_kde

durations_root = 'data/tungsten recordings/durations/'
spikefiles_dur = sorted(get_filenames(durations_root, 'SPK'))
stimfiles_dur = sorted(get_filenames(durations_root, 'STIM'))
due = DurationExperiment(spikefiles_dur, stimfiles_dur)

def filter_expon(varr, tau, dt, shape=1.):
        tarr_filter = np.arange(0, tau*15, dt)
        
        filt = np.concatenate([np.zeros(int(0e-3/dt)), dt*stats.gamma.pdf(tarr_filter, shape, scale=tau)])
        filtered_varr = np.convolve(varr, filt, mode='full')[:-len(filt)+1]
        return filtered_varr
    
durations = ['3ms','5ms','10ms','20ms','50ms','100ms','200ms','500ms','1000ms','2000ms','5000ms']

def get_stim_arr(dur_str, tau=0, peak=None, pre=1, mean_peak=250):
    kdes = []
    
    for resp in due[dur_str]:
        tarr = np.arange(-2, 8, 1e-4)
        kdes.append(resp.kde(tarr + resp.stim_start, bw=0.01))

    mean_kde = np.mean(kdes, axis=0)
    stim_arr = np.concatenate([np.ones(0*pre)*1.28, mean_kde])
    
    if peak is not None:
        stim_arr = peak * stim_arr / mean_peak
    
    if tau == 0:
        return stim_arr
    else:
        dt = 1e-4
        return filter_expon(stim_arr, tau, dt)
    
def expsyn(name):
    return f"""
    dg_syn/dt = -g_syn/tau{name} : Hz
    g{name}_post = g_syn : Hz (summed)
    """

def simulate_AL(duration, filtering=None, norn=10000, seed_num=42):
    start_scope()
    seed(seed_num)
    np.random.seed(seed_num)
    defaultclock.dt = 0.1*ms

    taustim = 2*ms
    tauexc = taustim
    tauinh = taustim
    tauslow = 750*ms
    
    if filtering is not None:
        stim_arr = filter_expon(get_stim_arr(duration, peak=150), filtering, 1e-4) + 30
    else:
        stim_arr = get_stim_arr(duration, peak=250) + 30
        
    firingrate = TimedArray(np.array(stim_arr)*Hz, dt=0.1*ms)

    eqstr = f"""
    dV/dt = -(V-EL)/taum - gSK*(V-ESK) - gstim*(V-Estim) - gexc*(V-Eexc) - ginh*(V-Einh) - gslow*(V-Einh) : 1 (unless refractory)
    gexc : Hz
    gslow : Hz
    ginh : Hz
    gstim : Hz

    dgSK/dt = -(gSK-xSK) / tauSKrise : Hz
    dxSK/dt = -xSK / tauSK : Hz
    SK_step : 1
    """

    eqsPN = Equations(eqstr, taum=20*ms, EL=0, Eexc=14/3, Estim=14/3, Einh=-2/3, ESK=-2/3,
                      tauSK=250*ms, tauSKrise=25*ms)

    eqsLN = Equations(eqstr, taum=20*ms, EL=0, Eexc=14/3, Estim=14/3, Einh=-2/3, ESK=-2/3,
                      tauSK=250*ms, tauSKrise=25*ms)

    reset_PN = """
    V = 0
    xSK += SK_step/(250*ms)
    """


    PN = NeuronGroup(10, eqsPN, method='exponential_euler', threshold='V>1', reset=reset_PN, refractory=2*ms)
    LN = NeuronGroup(6, eqsPN, method='exponential_euler', threshold='V>1', reset='V = 0', refractory=2*ms)

    PN.V = np.random.rand(10)
    LN.V = np.random.rand(6)

    PN.SK_step = np.clip(np.random.randn(10)*0.2+0.5, a_min=0, a_max=None)
    # PN.SK_step = 0
    LN.SK_step = 0

    ORN = PoissonGroup(norn, 'firingrate(t)')

    PN_input = Synapses(ORN, PN, model=expsyn('stim'), on_pre="g_syn += 0.004/taustim", method='exponential_euler')
    LN_input = Synapses(ORN, LN, model=expsyn('stim'), on_pre="g_syn += 0.0031/taustim", method='exponential_euler')
    S_EE = Synapses(PN, PN, model=expsyn('exc'), on_pre="g_syn += 0.01/tauexc", method='exponential_euler')
    S_EI = Synapses(PN, LN, model=expsyn('exc'), on_pre="g_syn += 0.01/tauexc", method='exponential_euler')
    S_IE = Synapses(LN, PN, model=expsyn('inh'), on_pre="g_syn += 0.0169/tauinh", method='exponential_euler')
    S_II = Synapses(LN, LN, model=expsyn('inh'), on_pre="g_syn += 0.015/tauinh", method='exponential_euler')
    S_IE_slow = Synapses(LN, PN, model=expsyn('slow'), on_pre="g_syn += 0.0338/tauslow", method='exponential_euler')
    S_II_slow = Synapses(LN, LN, model=expsyn('slow'), on_pre="g_syn += 0.04/tauslow", method='exponential_euler')

    PN_input.connect(p=0.01)
    LN_input.connect(p=0.01)
    S_EE.connect(p=0.75)  # 0.75
    S_EI.connect(p=0.75)  # 0.75
    S_IE.connect(p=0.38)  # 0.38
    S_IE_slow.connect(p=0.38)  # 0.38
    S_II.connect(p=0.25)  # 0.25
    S_II_slow.connect(p=0.25)  # 0.25

    variables = ['V', 'gstim', 'gSK', 'gslow', 'ginh', 'gexc']

    MV_PN = StateMonitor(PN, variables, record=True)
    MV_LN = StateMonitor(LN, variables, record=True)
    SM_PN = SpikeMonitor(PN)
    SM_LN = SpikeMonitor(LN)

    run(8*second)
    
    return MV_PN, MV_LN, SM_PN, SM_LN

def find_end(spiketrain, duration, start=2, min_duration=0.03):
    duration = max(duration, min_duration)
    isis = np.diff(spiketrain)
    end = (spiketrain[:-1][((spiketrain[1:] > start+duration) & (isis > 0.1))])[0]
    return end - start

def mean_end(SM, *args, **kwargs):
    tarr = np.arange(0, 8, 1e-3)

    ends = []

    for i, spike_train in SM.spike_trains().items():
        st = spike_train / second
        ends.append(find_end(st, *args, **kwargs))

    ends = np.array(ends)

    return ends.mean()


def run_AL_samples(trials=36, *args, **kwargs):
    '''
    run the <<run_network>> function on multiple cores
    '''
    
    def func(i):
        monitors = simulate_AL(*args, **kwargs, seed_num=i)
        return monitors[2].spike_trains()
        
    
    st_bundle = []
    
    for spike_trains in tqdm(Pool().uimap(func, range(trials)), total=trials):
        st_bundle.append(spike_trains)
    
    return st_bundle


if __name__ == '__main__':
#     res_complet = {}

#     for dur in durations:
#         print(dur)
#         res_complet[dur] = run_AL_samples(duration=dur, trials=36*20)
        
#     pickle.dump(res_complet, open( "data/generated/al_results.pickle", "wb" ) )
    res_complet = pickle.load( open( "data/generated/al_results.pickle", "rb" ) )
    
    kdes_trials = {}

    for dur in durations:
        kdes_trials[dur] = []

        for spiketrains in tqdm(res_complet[dur]):
            tarr = np.arange(0, 10, 1e-3)

            kdes = []

            for i, spike_train in spiketrains.items():
                st = spike_train / second
                kdes.append(get_kde(tarr, st, bw=0.01))

            kde = np.mean(kdes, axis=0)
            kdes_trials[dur].append(kde)
            
    pickle.dump(kdes_trials, open( "data/generated/al_kdes.pickle", "wb" ) )