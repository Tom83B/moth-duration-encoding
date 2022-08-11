import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import neo
from neo.io import AxonIO
from scipy.signal import butter, lfilter, freqz, savgol_filter, filtfilt
from scipy.stats import norm
import functools
from scipy.optimize import minimize

from utils import butter_lowpass_filter

class SpikeRecording:
    def __init__(self, spiketrain, start, end):
        self.spiketrain = spiketrain
        self.isis = np.diff(spiketrain)
        self.start = start
        self.end = end
        self.spont = None
    
    def segment(self, start, end):
        mask = (self.spiketrain >= start) & (self.spiketrain < end)
        return self.__class__(self.spiketrain[mask], start=start, end=end)
    
    def fit_spont(self):
        bursts = pp.RenewalProcess(pp.InvGaussDensity(), pp.ConstantIntensity(init=50))
        ibis = pp.RenewalProcess(pp.GammaDensity(), pp.ConstantIntensity(init=0.1))

        refrac = 2e-3

        spont = pp.MixedProcess(bursts, ibis, deadtime=refrac)

        spont.fit(self.spiketrain-self.spiketrain.min(), self.end-self.start)
        self.spont = spont
    
    def threshold(self):
        tmp_st = self.spiketrain
#         return 0.05
        return 10 ** threshold_otsu(np.log10(self.isis))
    
    def burst_ratio(self):
        return (self.isis < self.threshold()).mean()
    
    def burst_times(self):
        return self.spiketrain[1:][self.isis >= self.threshold()]
    
    def burst_lengths(self):
        ixs = np.argwhere(self.isis >= self.threshold()).flatten()
        bls = np.diff(ixs)
        return bls
        

class Rebound(SpikeRecording):
    def __init__(self, spiketrain, start, end, fit_type='double'):
        self.fit_type = fit_type
        self.spiketrain = spiketrain
        self.isis = np.diff(self.spiketrain)
        self.start = start
        self.end = end
#         self.bursttrain = self.spiketrain[1:][self.isis > self.otsu_threshold()]
        self.fitted = False
    
    @property
    def bursttrain(self):
        return self.spiketrain[1:][self.isis > self.otsu_threshold()]
    
    @classmethod
    def from_segment(cls, segment, fit_type='double'):
        return cls(segment.spiketrain, segment.start, segment.end, fit_type)
    
    def fit_point_process(self):
        if not self.fitted:
            A1, tau1, A2, tau2, B = self.rebound_params()
            q = (self.isis > self.otsu_threshold()).mean()
            rate_burst = 1 / self.isis[self.isis < self.otsu_threshold()].mean()
            x0 = (A1, A2, B, tau1, tau2, q, rate_burst)
            st = self.spiketrain - self.start
            self.tottime = self.end - self.start
            self.params = find_opt_params(x0, st, self.tottime)
            self.fitted = True
            self.mproc = MixedProcess(*self.params, self.tottime)
    
    def threshold(self):
        if not self.fitted:
            self.fit_point_process()
        st = self.spiketrain - self.start
        return self.mproc.threshold(st[:-1], perc=0.3)
    
    @functools.lru_cache()
    def otsu_threshold(self):
        isi_count_left = 20
        isi_count_right = 20
        isi_count_tot = isi_count_left + isi_count_right
        
        if len(self.isis) > isi_count_tot:
            threshold_vals = []
            for i in range(isi_count_left, len(self.isis)-isi_count_right):
                instathresh = 10 ** threshold_otsu(np.log10(self.isis[i-isi_count_left:i+isi_count_right]))
                threshold_vals.append(instathresh)

            threshold_vals = [threshold_vals[0]]*isi_count_left + threshold_vals + [threshold_vals[-1]]*isi_count_right
            return np.array(threshold_vals)
        else:
            thr = 10 ** threshold_otsu(np.log10(self.isis))
            return np.ones_like(self.isis) * thr
    
    def rate_func(self, t, *params):
        if self.fit_type == 'double':
            a1, b1, a2, b2, c = params
            return a1*np.exp(-t/b1) + a2*np.exp(-t/b2) + c
        else:
            a, b, c = params
            return a*np.exp(-t/b) + c
       
    def mean_total(self, *params):
        duration = self.end - self.start
        if self.fit_type == 'double':
            a1, b1, a2, b2, c = params
            return b1*a1*(1-np.exp(-duration/b1)) + b2*a2*(1-np.exp(-duration/b2)) + c*duration
        else:
            a, b, c = params
            return b*a*(1-np.exp(-duration/b)) + c*duration
    
    @functools.lru_cache()
    def binarize_burst_train(self, dt):
        bins = np.arange(0, self.end-self.start, dt)
        counts, _ = np.histogram(self.bursttrain-self.start, bins=bins)
        centers = (bins[1:] + bins[:-1]) / 2
        return centers, counts
        
    def get_loglik(self, params):
        dt = 0.01
#         bins = np.arange(0, self.end-self.start, dt)
#         counts, _ = np.histogram(spiketrain-self.start, bins=bins)
#         centers = (bins[1:] + bins[:-1]) / 2

        logint = np.log(self.rate_func(self.bursttrain-self.start, *params))
#         print(logint.sum())
        loglik = logint.sum() - self.mean_total(*params)
        return loglik

#         centers, counts = self.binarize_burst_train(dt)
#         rate_func = self.rate_func(centers, *params)

#         spike_mask = counts > 0
#         loglik = np.log(rate_func[spike_mask]*dt).sum() + np.log(1-rate_func[~spike_mask]*dt).sum()
#         return loglik
    
    @functools.lru_cache()
    def rebound_params(self):
#         bstarts_rebound = self.bursttrain - self.start
        func = lambda x: -self.get_loglik(x)
        
        if self.fit_type == 'double':
            x0 = (5, 70, 20, 10, 0.05)
            bounds = [(.0, 20), (1, 5000), (0, 20), (2, 100), (0, 5)]
        else:
            x0 = (5, 70, 0.05)
            bounds = [(.1, 20), (1, 1000), (0, 5)]
            
        res = minimize(func, x0=x0, bounds=bounds)
        return res.x
    
    def plot_isis(self, ax, start=0, **kwargs):
        xx = np.arange(len(self.isis)) + start
        tt = self.spiketrain[1:]
#         rates = self.rate_func(tt-self.start, *self.rebound_params())
        rates = self.mproc.rate_func(tt-self.start)
        ax.plot(xx, 1/rates, **kwargs)
        ax.set_ylabel('ISI (s)')
    
class LFPRecording:
    def __init__(self, lfp, start=0, dt=1e-4):
        self._lfp = lfp
        self.dt = dt
        self._tarr = np.arange(len(lfp)) * dt + start
    
    def segment(self, start, end):
        mask = (self._tarr >= start) & (self._tarr < end)
        return self.__class__(self._lfp[mask], start=start, dt=self.dt)
    
    def to_arrays(self):
        return self._tarr, self._lfp
    
    @classmethod
    def from_abf(cls, filename, filtered=True, sig=0, cutoff=50, downsample=False):
        reader = AxonIO(filename=filename)
        seg = reader.read_segment()
        dt = seg.analogsignals[0].sampling_period.base.item()
        if filtered:
#             if downsample and abs(dt-1e-4)<1e-5:
#                 signal = np.array(seg.analogsignals[sig])[:,0][::4]
#                 dt = dt * 4
#             else:
#                 signal = np.array(seg.analogsignals[sig])[:,0]
                
            lfp = butter_lowpass_filter(np.array(seg.analogsignals[sig])[:,0], cutoff, int(1/dt))
#             bw = 50 # 100
#             lfp = savgol_filter(signal, int(2*(np.round(bw*0.0005/dt)//2)+1), 3)
        else:
            lfp = np.array(seg.analogsignals[sig])[:,0]
        return cls(lfp, dt=dt)

class LFPResponse:
    def __init__(self, lfp, stim_start, stim_end, dt=1e-4):
        self._lfp = lfp
        self.start = stim_start
        self.end = stim_end
    
    def to_arrays(self, before=1, after=8):
        tstart = self.start - before
        tend = self.end + after
        return self._lfp.segment(tstart, tend).to_arrays()
    
    def plot(self, ax, before=1, after=8, **kwargs):
        tarr, varr = self.to_arrays(before, after)
        ax.plot(tarr, varr, **kwargs)

class SpikingResponse:
    def __init__(self, rec, stim_start, stim_end, dose=None):
        self._rec = rec
        offset = 0.1
        reb_st = rec.segment(stim_end + offset, rec.end).spiketrain
        self.rebound = Rebound(reb_st, stim_end+offset, rec.end)
        self.stim_start = stim_start
        self.stim_end = stim_end
        self.dose = dose
        
    def fit_status(self):
        if self.rebound.fitted:
            return 'Y'
        elif self.eligible():
            return 'N'
        else:
            return '0'
    
    @classmethod
    def from_files(cls, spike_times, valves, dose=None):
        spiketrain = pd.read_csv(spike_times)['spike times'].values
        try:
            rec = SpikeRecording(spiketrain, 0, spiketrain.max())
        except:
            rec = SpikeRecording(spiketrain, 0, 0)
        valve_times = pd.read_csv(valves, sep='\t').values[0]
#         rid = spike_times.split('/')[-1].split('.')[0]
        return cls(rec, *valve_times, dose)
    
    @property
    def spiketrain(self):
        return self._rec.spiketrain
    
    @property
    def isis(self):
        return np.diff(self._rec.spiketrain)
    
    @property
    def duration(self):
        dur_exact = self.stim_end - self.stim_start
        exponent = np.floor(np.log10(dur_exact))
        dec = 10 ** exponent
        return np.round(dur_exact / dec) * dec
    
    def fit(self):
        self.rebound.fit_point_process()
    
    def threshold(self):
        self.fit()
        return self.rebound.threshold()[0]
#         tmp_st = self.spiketrain[(self.spiketrain > self.stim_end)][:]
#         return 10 ** threshold_otsu(np.log10(self.isis))
#         return 0.1

    def intensity(self):
        return ((self.spiketrain > self.stim_start) & (self.spiketrain <= self.stim_start+0.2)).sum()

    def _after_response_ibi_mask(self):
        isi_starts = self.spiketrain[:-1]
        isi_ends = self.spiketrain[1:]
        after_response_mask = (isi_ends > self.stim_end) & (isi_starts > self.stim_start)
        above_thr_mask = self.isis > self.threshold()
        
        return after_response_mask & above_thr_mask
    
    def eor(self):
        try:
            ix = np.argwhere(self._after_response_ibi_mask()).flatten()[0]
            return ix
        except:
            return None
    
    def sor(self):
        try:
            ix = np.argwhere(self.spiketrain > self.stim_start).flatten()[0]
            return ix
        except:
            return None
    
#     def _number_of_response_spikes(self):
#         return self.eor() - self.sor()
    
#     def _number_of_rebound_spikes(self):
#         return len(self.rebound.spiketrain)
    
    def eligible(self, resp_cond=5, reb_cond=20):
        eligible_response = ((self.spiketrain > self.stim_start) & (self.spiketrain <= self.stim_start+0.1)).sum() >= resp_cond
        eligible_rebound = (self.spiketrain > self.stim_end).sum() >= reb_cond
        return eligible_response & eligible_rebound
    
    def rebound_isi(self):
        return np.exp(np.log(self.isis[self._after_response_ibi_mask()][1:11]).mean())
    
    def gap(self):
        ix = self.eor()
        if ix is not None:
            return self.isis[ix]
        else:
            return np.nan
    
    def sample_diffs(self, n=50000):
        st = self.spiketrain[self.eor()+1:]
        st_start = st.min()
        st = st - st_start
        rebound_params = find_opt_params(x0=self.rebound.params, spiketrain=st, tottime=st.max())
        mproc = MixedProcess(*rebound_params, st.max())
        last_spike = self.spiketrain[self.eor()] - st_start
        ibis = mproc.generate_ibis(last_spike, n) - last_spike
        return self.gap()-ibis
    
    def kde(self, tarr, align=True, bw=0.1, shift=False):
        rate = np.zeros_like(tarr)
        
        if shift:
            spiketrain = self.spiketrain - self.stim_start
        else:
            spiketrain = self.spiketrain
        
        st_tmp = spiketrain[(spiketrain >= tarr.min()-5) & (spiketrain < tarr.max()+5)]
#         est = norm.pdf(tarr[:,None], loc=st_tmp[None,:], scale=bw).sum(axis=1)

        if not callable(bw):
            est = norm.pdf(tarr, loc=st_tmp[:,None], scale=bw).sum(axis=0)
        else:
            est = norm.pdf(tarr, loc=st_tmp[:,None], scale=bw(st_tmp)[:,None]).sum(axis=0)
        
        if align:
            tarr = tarr-tarr[0]
        return est
    
    def plot(self, ax):
        xx = np.arange(len(self.isis))
        isi_starts = self.spiketrain[:-1]
        mask = (isi_starts <= self.stim_start) | (isi_starts > self.stim_end)
        ax.scatter(xx[mask], self.isis[mask], s=5)
        ax.scatter(xx[~mask], self.isis[~mask], s=5)
        
        xx_thr = np.arange(len(self.rebound.isis)) + 1 + self.eor()
        ax.plot(xx_thr, self.rebound.threshold(), color='black', linestyle='dashed')
#         ax.axhline(self.threshold(), color='black', linestyle='dashed')
        
        ix = self.eor()
        if ix is not None:
            ax.scatter(xx[[ix]], [self.isis[ix]], color='red', s=10)
        ax.set_yscale('log')
        ax.set_xlabel('ISI index')
        ax.set_ylabel('ISI (s)')

class StabilityRecording:
    def __init__(self, txt_file, abf_file=None, recording_duration=1800, offset=30, pauses=300, filtered=True, sig=0, cutoff=50, downsample=False):
        self.name = txt_file.split('/')[-1].split('.')[0]
        txt_data = pd.read_csv(txt_file, sep='\t')
        spiketrain = txt_data['spike times'].values
        self._spikerec = SpikeRecording(spiketrain, 0, spiketrain.max())
        self.spiketrain = self._spikerec.spiketrain
        
        if 'Vanne1 ON' in txt_data.columns:
            self._valve_times = txt_data[['Vanne1 ON', 'Vanne1 OFF']].dropna().values
        else:
            self._valve_times = np.array([[offset+s*300, offset+s*300+2] for s in range(6)])
        
        if abf_file is not None:
            self._lfp = LFPRecording.from_abf(abf_file, filtered=filtered, sig=sig, cutoff=cutoff, downsample=downsample)
        else:
            self._lfp = None
        
#         self.first_stim = txt_data['Vanne1 ON'].values[0]
        
        self._lfp_responses = []
        self._spike_responses = []
        
        old_voffs = np.concatenate(([0], self._valve_times[:-1,1]))
        next_vons = np.concatenate((self._valve_times[1:,0], [recording_duration]))
        
        for old_voff, (von, voff), next_von in zip(old_voffs, self._valve_times, next_vons):
            if self._lfp is not None:
                self._lfp_responses.append(LFPResponse(self._lfp, von, voff))
#             tmp_st = self.spiketrain[(self.spiketrain > old_voff) & (self.spiketrain < next_von)]
            sr = self._spikerec.segment(old_voff, next_von)
            self._spike_responses.append(SpikingResponse(sr, von, voff))
#             old_voff = voff
    
    def __len__(self):
        return len(self._spike_responses)
    
    def __getitem__(self, arg):
        if self._lfp is not None:
            return self._spike_responses[arg], self._lfp_responses[arg]
        else:
            return self._spike_responses[arg]

class StabilityDurationRecording(StabilityRecording):
    def __init__(self, txt_file, abf_file=None, filtered=True, sig=1, cutoff=50, downsample=False):
        txt_data = pd.read_csv(txt_file, sep='\t')
        spiketrain = txt_data['spike times'].values
        recording_duration = spiketrain.max() + 10
        super().__init__(txt_file, abf_file, recording_duration=recording_duration, filtered=filtered, sig=1, cutoff=cutoff, downsample=downsample)
        
        self._responses = {}
        
        for i in range(len(self._spike_responses)):
            sresp = self._spike_responses[i]
            if abf_file is not None:
                lresp = self._lfp_responses[i]
            dur = sresp.duration
            
            if dur not in self._responses:
                self._responses[dur] = []
                
            if abf_file is not None:
                self._responses[dur].append((sresp, lresp))
            else:
                self._responses[dur].append(sresp)
    
    def __getitem__(self, arg):
        return self._responses[arg]
    
    

class DoseDurationRecording:
    def __init__(self, spike_times, valves):
        self.spiketrain = pd.read_csv(spike_times)['spike times'].values
        self.valve_times = pd.read_csv(valves, sep='\t').values
        self.rid = spike_times.split('/')[-1].split('.')[0]
        
        responses = []
        keys = []
        
        stim_ends = np.concatenate(([0], self.valve_times[:-1,1]))
#         stim_starts = np.concatenate((self.valve_times[1:,0], [10000]))
        stim_starts = np.concatenate((self.valve_times[1:,0], [self.spiketrain.max()]))
        
        for i, (se, ss, v) in enumerate(zip(stim_ends, stim_starts, self.valve_times)):
            st_tmp = self.spiketrain[(self.spiketrain > se) & (self.spiketrain <= ss)]
            dose = self._dose(i)
            rec = SpikeRecording(st_tmp, se, ss)
            resp = SpikingResponse(rec, *v, dose)
#             response = Response(st_tmp, *v, dose, self.rid)
            responses.append(resp)
            dur_str = f'{resp.duration*1000:.0f}ms'
            keys.append((dose, dur_str))
        
        self._responses = {key: resp for key, resp in zip(keys, responses)}

    def __getitem__(self, key):
        dose, duration = key
        
        try:
            return self._responses[(dose, duration)]
        except KeyError:
            return None
    
    def fit(self):
        keys = list(self._responses.keys())
        
        responses = []
        for key in self._responses.keys():
            if self._responses[key].eligible():
                self._responses[key].fit()
        
    def _dose(self, i):
        if i < 3:
            return '10pg'
        elif i < 6:
            return '100pg'
        else:
            return '1ng'

class DoseDurationExperiment:
    def __init__(self, spike_files, valve_files):
        self._recordings = []
        
        for sfile, vfile in zip(spike_files, valve_files):
            self._recordings.append(DoseDurationRecording(sfile, vfile))
    
    def fit_recordings(self):
        self._recordings = fit_group(self._recordings, check_eligibility=False)
    
    def __getitem__(self, key):
        recordings = []
        for r in self._recordings:
            resp = r[key]
            if resp is not None:
                recordings.append(r[key])
        return recordings

class DurationExperiment:
    def __init__(self, spike_files, valve_files):
        self._responses = {}
        
        for sfile, vfile in zip(spike_files, valve_files):
            response = SpikingResponse.from_files(sfile, vfile, dose='100pg')
            dur_str = f'{response.duration*1000:.0f}ms'
            self._responses[dur_str] = self._responses.get(dur_str, list()) + [response]
    
    def fit_responses(self):
        for dur, resp_list in self._responses.items():
            self._responses[dur] = fit_group(resp_list)
    
    def __getitem__(self, key):
        
        try:
            return self._responses[key]
        except KeyError:
            return None
        
        
def fit_response(ix, resp):
    resp.fit()
    return ix, resp

def fit_group(group, check_eligibility=True):
    ixs = []
    
    if check_eligibility:
        filtered_group = [x for x in group if x.eligible()]
        filtered_ixs = [ix for ix, x in enumerate(group) if x.eligible()]
    else:
        filtered_group = [x for x in group]
        filtered_ixs = [ix for ix, x in enumerate(group)]
    
    for ix, resp in tqdm(Pool().uimap(fit_response, filtered_ixs, filtered_group), total=len(filtered_group), smoothing=0):
        group[ix] = resp
    
    return group