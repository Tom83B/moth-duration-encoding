import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def deriv_matrix(O, ka, sa, kb, sb):
    return np.array([
        [-(O*kb*sb)  , sb            , 0   ],
        [O*kb*sb     , -(sb+ka*sa), sa  ],
        [0           , ka*sa      , -sa ]
    ])

def deriv_receptor(t, x, O_func, *args):
    return deriv_matrix(O_func(t), *args)@x

def deriv_LFP(t, x, O_func, *args):
    *recept, lfp = x
    
    drdt = deriv_matrix(O_func(t), *args)@recept
    dlfpdt = -(lfp-recept[2])/1.e-2
    return *drdt, dlfpdt


def get_stim_func(dur, conc=1, lag=5e-3):
    def func(t):
        return np.heaviside(t-lag, 0)*(1-np.heaviside(t-dur-lag, 0))*conc
    
    return func

def error_LFP(params, varrs):
    max_step = 1e-3
    t_min = -0.1
    t_max = 0.4
    tarr = np.linspace(t_min, t_max, 500)

    deriv_params = params
    
    args = (get_stim_func(0.2), *deriv_params)
    y0 = np.array([1, 0, 0, 0])
    res = solve_ivp(deriv_LFP, t_span=(t_min,t_max), y0=y0, args=args, t_eval=tarr, max_step=max_step)
    rescaling = varrs[1].max() / res.y[-1].max()
    err_med = ((res.y[-1]*rescaling - varrs[1])**2).sum()
    
    args = (get_stim_func(0.02), *deriv_params)
    y0 = np.array([1, 0, 0, 0])
    res = solve_ivp(deriv_LFP, t_span=(t_min,t_max), y0=y0, args=args, t_eval=tarr, max_step=max_step)
    err_fast = ((res.y[-1]*rescaling - varrs[0])**2).sum()
    
    
#     print(params, err_med+err_fast)
    return err_med+err_fast

def optimize_params(varrs, ka=7, sa=7, kb=40, sb=130):
    x0 = (ka, sa, kb, sb)
    res_min = minimize(error_LFP, x0, args=(varrs,), tol=1e-2, bounds=[
        (ka/10, None), (sa/10, None), (kb/10, None), (sb/10, None)
    ])
    return res_min