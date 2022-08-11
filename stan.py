import numpy as np


def moving_average(a, n=30) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def eligible(spiketrain, dur=0):
    if (((spiketrain > 0) & (spiketrain < 0.1)).sum() > 5)  & ((spiketrain > dur).sum() >= 1):
        return True
    else:
        return False

def find_end(spiketrain, dur, min_gap=0.1):
    if eligible(spiketrain, dur):
        isis = np.diff(spiketrain)
        mask = (spiketrain[:-1] > 0) & (spiketrain[1:] > dur) & (isis > min_gap)
        
        if mask.sum() > 0:
            return spiketrain[np.argwhere(mask).flatten()[0]]
        else:
            return np.nan
    else:
        return np.nan

def apply_func(responses, func, *args, **kwargs):
    return np.array([func(np.append(resp.spiketrain-resp.stim_start, np.inf), *args, **kwargs) for resp in responses])

def bw_func(tt, tau=0.5, min_bw=0.005, max_bw=0.1):
    bw_diff = max_bw - min_bw
    scales = np.clip(bw_diff*(1-np.exp(-tt/tau))+min_bw, a_min=min_bw, a_max=max_bw)
    return scales