import numpy as np
from scipy.interpolate import UnivariateSpline

from loaders import StabilityDurationRecording
from utils import get_filenames


def intact_files(leave_out=None):
    txt_files1 = sorted(get_filenames('data/base recordings/single trial','txt'))
    abf_files1 = sorted(get_filenames('data/base recordings/single trial','abf'))
    
    txt_files2 = sorted(get_filenames('data/base recordings/multiple trials','txt'))
    abf_files2 = sorted(get_filenames('data/base recordings/multiple trials','abf'))

    txt_files = txt_files2 + txt_files1
    abf_files = abf_files2 + abf_files1

    inhibited = []

    for file_ix in range(len(txt_files)):
        srec, lrec = StabilityDurationRecording(txt_files[file_ix], abf_files[file_ix], filtered=False)[2][1]
        st = srec.spiketrain - srec.stim_start

        if ((st > 2.05) & (st < 2.5)).sum() == 0:
            inhibited.append(file_ix)
    
    # leave out recordings which are too noisy or contain artifacts in LFP during stimulation
    if leave_out is None:
        leave_out = []
    
    intact_txt = [txt_files[ix] for ix in inhibited if ix not in leave_out]
    intact_abf = [abf_files[ix] for ix in inhibited if ix not in leave_out]
    
    return intact_txt, intact_abf

def load_base(bw=0.0075, cutoff=200, leave_out=None, span=(-1,3), dt=1e-3):
    intact_txt, intact_abf = intact_files(leave_out)
    
    lfps = {}
    kdes = {}
    tarrs = {}

    for dur in [0.02, 0.2, 2]:
        lfps[dur] = []
        kdes[dur] = []

    kde_xx = np.arange(span[0], span[1], dt)

    for txt_file, abf_file in zip(intact_txt, intact_abf):
        rec = StabilityDurationRecording(txt_file, abf_file,
                                         filtered=True, cutoff=cutoff)

        for i in range(1):
            for dur in [0.02, 0.2, 2]:
                sresp, lresp = rec[dur][i]

                tarr, varr = lresp.to_arrays(before=3)
                tarr = tarr - sresp.stim_start
                varr = varr - varr[tarr < 0].mean()

                func = UnivariateSpline(tarr, varr, s=0, k=1)

                lfps[dur].append(func(kde_xx))

                kde = sresp.kde(kde_xx, shift=True, bw=bw)
                kdes[dur].append(kde)

    return kdes, lfps, kde_xx

def get_means(bw=0.0075, cutoff=200, leave_out=None, span=(-1,3), dt=1e-3):
    kdes, lfps, kde_xx = load_base(bw, cutoff, leave_out, span, dt)
    
    if leave_out is None:
        bad_ix = {
            0.02: [9],
            0.2: [],
            2: []
        }
    else:
        bad_ix = {
            0.02: [],
            0.2: [],
            2: []
        }

    mean_lfps = {}
    mean_kdes = {}

    for dur in [0.02, 0.2, 2]:
        mean_lfps[dur] = np.mean([lfps[dur][ix] for ix in range(len(lfps[dur]))
                                  if ix not in bad_ix[dur]], axis=0)
        mean_kdes[dur] = np.mean([kdes[dur][ix] for ix in range(len(kdes[dur]))
                                  if ix not in bad_ix[dur]], axis=0)
        
    return mean_kdes, mean_lfps, kde_xx