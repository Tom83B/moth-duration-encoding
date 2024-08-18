import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pathos.multiprocessing import ProcessingPool as Pool

from loaders import StabilityDurationRecording
from lfpan import intact_files
from utils import resample, filter_expon, filter_from_coefs

    
def fit_model(txt_file, abf_file, taus, shapes):
    tarr_universal = np.arange(-1,8,0.002)
    rec = StabilityDurationRecording(txt_file, abf_file, filtered=True, downsample=True, cutoff=15, sig=0)
    
    X = {}
    y = {}
    kdes = {}
    tarrs = {}
    varrs = {}
    ends = {}
    preds = {}

    for dur in [0.02, 0.2, 2]:
        sresp, lresp = rec[dur][0]

        tarr, varr = lresp.to_arrays(before=3)
        dt = tarr[1] - tarr[0]
        tarr = tarr - sresp.stim_start
        varr = varr - varr[tarr < 0].mean()
        kde = sresp.kde(tarr, shift=True, bw=0.015)

        spiketrain = sresp.spiketrain - sresp.stim_start

        start = -3
        end = 8
        resp_mask = (tarr > start) & (tarr < end)
        
        kdes[dur] = resample(tarr[resp_mask], kde[resp_mask], tarr_universal)
        tarrs[dur] = tarr[resp_mask][:]
        varrs[dur] = varr[resp_mask]

        X_tmp = np.array([filter_expon(varr, tau, shape, dt) for tau, shape in zip(taus, shapes)]).T[resp_mask]

        dt = tarr[1] - tarr[0]
        spike_mask = np.zeros(X_tmp.shape[0], dtype=bool)
        spike_mask[((spiketrain[(spiketrain > start) & (spiketrain < end)] - start) / dt).astype(int)] = True
        y_tmp = kde[resp_mask]

        X[dur] = X_tmp[:]
        y[dur] = y_tmp[:]

    dt = tarr[1] - tarr[0]

    reg = LinearRegression(fit_intercept=False)

    resp_mask_train = (tarrs[dur] > -1) & (tarrs[dur] < 2)
    X_all = X[2][resp_mask_train]
    y_all = y[2][resp_mask_train]
    
    weights = np.concatenate([
        np.ones_like(y[dur]) * 1 for dur in [0.02, 0.2, 2]
    ])
    reg.fit(X_all, y_all)
    
    filt = filter_from_coefs(reg.coef_, taus, shapes, dt)

    for dur in [0.02, 0.2, 2]:
        resp_tarr = tarrs[dur]
        
        yhat = reg.predict(X[dur])
        
        preds[dur] = yhat
    
    return ends, kdes, preds, tarr_universal, filt, varrs, reg.coef_

def run_all(taus, shapes, filename):
    ends_all = []
    kdes_all = []
    preds_all = []
    filts_all = []
    varrs_all = []
    coefs_all = []
    
    for txt_file, abf_file in tqdm(zip(intact_txt, intact_abf), total=len(intact_txt)):
        ends, kdes, preds, tarr_universal, filt, varr, filt_coefs = fit_model(txt_file, abf_file, taus, shapes)
        ends_all.append(ends)
        kdes_all.append(kdes)
        preds_all.append(preds)
        filts_all.append(filt)
        varrs_all.append(varr)
        coefs_all.append(filt_coefs)
        
    d = {
        'ends_all': ends_all,
        'kdes_all': kdes_all,
        'preds_all': preds_all,
        'filts_all': filts_all,
        'varrs_all': varrs_all,
        'coefs_all': coefs_all
    }
  
    with open(f'data/generated/{filename}.pkl', 'wb') as file:
        pickle.dump(d, file)

if __name__ == '__main__':
    # load files and remove too noisy recordings and recordings with artifacts
    intact_txt, intact_abf = intact_files(leave_out=[7,11,17,32])
    
    shapes = [1, 1, 1]
    taus_all = [0, 29e-3, 0.628]
    taus_all = [0, 31e-3, 0.635]
    
    run_all(taus_all, shapes, 'full')
    run_all(taus_all[:-1], shapes[:-1], 'reduced')