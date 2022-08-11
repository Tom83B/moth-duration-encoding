from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.stats import gamma, norm
from scipy.interpolate import UnivariateSpline
import numpy as np
import os

def butter_lowpass(cutoff, fs, order=5, btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1, btype='low'):
    b, a = butter_lowpass(cutoff, fs, order=order, btype=btype)
    y = filtfilt(b, a, data)
    return y

def despine_ax(ax, where=None, remove_ticks=None):
    if where is None:
        where = 'trlb'
    if remove_ticks is None:
        remove_ticks = where
    
    if remove_ticks is not None:
        if 'b' in where:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if 'l' in where:
            ax.set_yticks([])
            ax.set_yticklabels([])
    
    to_despine = []
    
    if 'r' in where:
        to_despine.append('right')
    if 't' in where:
        to_despine.append('top')
    if 'l' in where:
        to_despine.append('left')
    if 'b' in where:
        to_despine.append('bottom')
    
    for side in to_despine:
        ax.spines[side].set_visible(False)
        
def plot_scalebar(ax, start, vertical=None, horizontal=None):
    sb_kwargs = dict(color='black', lw=3)
    
    if vertical is not None:
        ax.plot([start[0], start[0]], [start[1], start[1]+vertical['size']], **sb_kwargs)
        ax.text(start[0]+vertical['x'], start[1]+vertical['y'], vertical['text'])
    
    if horizontal is not None:
        ax.plot([start[0], start[0]+horizontal['size']], [start[1], start[1]], **sb_kwargs)
        ax.text(start[0]+horizontal['x'], start[1]+horizontal['y'], horizontal['text'])
        
def get_filenames(folder, contains=''):
    if folder[-1] != '/':
        folder = folder + '/'
        
    return [folder+file for file in os.listdir(folder) if contains in file]

def signif(pval):
    if pval < 1e-3:
        return '***'
    elif pval < 1e-2:
        return '**'
    elif pval < 5e-2:
        return '*'
    else:
        return ''
            
def resample(tarr, arr, new_tarr):
    f = UnivariateSpline(tarr, arr, s=0, k=1)
    return f(new_tarr)

def filter_expon(varr, tau, shape, dt):
        tarr_filter = np.arange(0, tau*15, dt)
        
        if tau > 0:
            filt = np.concatenate([np.zeros(int(0e-3/dt)), gamma.pdf(tarr_filter, shape, scale=tau)*dt])
            filtered_varr = np.convolve(varr, filt, mode='full')[:-len(filt)+1]
            return filtered_varr
        else:
            return varr

def filter_from_coefs(coefs, taus, shapes, dt):
    tarr = np.arange(0, 5, dt)
    farr = np.zeros_like(tarr)
    
    for c, tau, shape in zip(coefs, taus, shapes):
        if tau > 0:
            farr += c*gamma.pdf(tarr, shape, scale=tau)*dt
    
    return tarr, farr

def get_kde(tarr, spiketrain, bw):
    return norm.pdf(tarr[:,None], loc=spiketrain[None,:], scale=bw).sum(axis=1)

def rescale(arr, ret_scale=False):
    if ret_scale == False:
        return arr / arr.max()
    else:
        return arr / arr.max(), arr.max()