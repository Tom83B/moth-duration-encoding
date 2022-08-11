Cmd = 3.28*1e-3
gld = 0.4373*1e-3
Cms = 1.44*1e-3
gls = 1.44*1e-3
gi = 2.011*1e-3
ge = 26.77*1e-3
Cma = 30*1e-3
ga = 3.1*1e-3
Eld = -97
Els = -62
Ea = -35
Er = 0

Vea = 35
Vis = -62
Ved = 35
Vid = -62
Ek = -90
Ir = 0

def deriv(y, t, Ir_func, Iad_func,
          Cma=30*1e-3, ga=3.1e-3,
          Cmd=3.28*1e-3, gld=0.4373*1e-3,
          Cms=1.44e-3*0.3, gls=1.44e-3*0.3,
          Eld=-97, Ea=-35, ge=26.77e-3, ret_Ir=False):
    Vea, Vis, Ved, Vid = y
    
    Ii = gi * (Vid - Vis)
    Ia = -ga * (Vea + Ea)
    Ie = ge * (Vea - Ved)
    Ild = gld * (Ved - Vid + Eld)
    Ils = gls * (Vis - Els)# + gls * (rect(Vis + 55)) * soma_rect
    
    dt = 0.001
    
    Ir = Ir_func(t)
    Iad = Iad_func(t)
    
    dVea = (Ia - Ie) / Cma
    dVis = (Ii - Ils - Iad) / Cms
    dVed = (gi / (Cmd*(ge+gi))) * (Ie - Ild - Ir) +\
           (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
           (gi / (Cms*(ge+gi))) * (Ii - Ils - Iad)
    dVid = -(ge / (Cmd*(ge+gi))) * (Ie - Ild - Ir) +\
            (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
            (gi / (Cms*(ge+gi))) * (Ii - Ils - Iad)
    
    if ret_Ir:
        return np.array([dVea, dVis, dVed, dVid, Ir, Ie, Ild, Ia, Ii, Ils])
    else:
        return np.array([dVea, dVis, dVed, dVid])
    
def deriv_known_LFP(t, y, LFP,
          Cma=30*1e-3, ga=3.1e-3,
          Cmd=3.28*1e-3, gld=0.4373*1e-3,
          Cms=1.44e-3, gls=1.44e-3,
          Eld=-97, Ea=-35, ret_Ir=False):
    Vea, Vis, Ved, Vid, *_ = y
    
    Ved = LFP(t)
    
    Ii = gi * (Vid - Vis)
    Ia = -ga * (Vea + Ea)
    Ie = ge * (Vea - Ved)
    Ild = gld * (Ved - Vid + Eld)
    Ils = gls * (Vis - Els)# + gls * (rect(Vis + 55)) * soma_rect
    
    dt = 0.1
    dVed = (LFP(t+dt) - LFP(t-dt)) / (2*dt)
    
    Ir = (Cmd*(ge+gi)/gi) * (-dVed + \
       (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
       (gi / (Cms*(ge+gi))) * (Ii - Ils)) + Ie-Ild
    
    dVea = (Ia - Ie) / Cma
    dVis = (Ii - Ils) / Cms
    dVed = (gi / (Cmd*(ge+gi))) * (Ie - Ild - Ir) +\
           (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
           (gi / (Cms*(ge+gi))) * (Ii - Ils)
    dVid = -(ge / (Cmd*(ge+gi))) * (Ie - Ild - Ir) +\
            (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
            (gi / (Cms*(ge+gi))) * (Ii - Ils)
    
    if ret_Ir:
        return np.array([dVea, dVis, dVed, dVid, Ir, Ie, Ild, Ia, Ii, Ils])
    else:
        return np.array([dVea, dVis, dVed, dVid])
    
def deriv_known_Vis(t, y, input_current, soma,
          Cma=30*1e-3, ga=3.1e-3,
          Cmd=3.28*1e-3, gld=0.4373*1e-3,
          Cms=1.44e-3, gls=1.44e-3,
          Eld=-97, Ea=-35, ret_Ir=False):
    Vea, Vis, Ved, Vid, *_ = y
    
    Vis = soma(t)
    
    Ii = gi * (Vid - Vis)
    Ia = -ga * (Vea + Ea)
    Ie = ge * (Vea - Ved)
    Ild = gld * (Ved - Vid + Eld)
    Ils = gls * (Vis - Els)# + gls * (rect(Vis + 55)) * soma_rect
    
    dt = 0.1
    dVis = (soma(t+dt) - soma(t-dt)) / (2*dt)
    
    Ir = input_current(t)
    Iad = -(Cms*dVis - Ii + Ils)
    
#     Ir = gr_func(t) * (Ved - Vid + 0)
#     Iad = gad_func(t) * (Vis - Ek)  # typically a positive current
#     Ir = Ir_func(t)
#     Iad = Iad_func(t)
    
    dVea = (Ia - Ie) / Cma
    dVis = (Ii - Ils - Iad) / Cms
    dVed = (gi / (Cmd*(ge+gi))) * (Ie - Ild - Ir) +\
           (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
           (gi / (Cms*(ge+gi))) * (Ii - Ils - Iad)
    dVid = -(ge / (Cmd*(ge+gi))) * (Ie - Ild - Ir) +\
            (ge / (Cma*(ge+gi))) * (Ia - Ie) +\
            (gi / (Cms*(ge+gi))) * (Ii - Ils - Iad)
    
    if ret_Ir:
        return np.array([dVea, dVis, dVed, dVid, Iad, Ie, Ild, Ia, Ii, Ils])
    else:
        return np.array([dVea, dVis, dVed, dVid])

def get_currents(y, aux, dend, soma):
    ga = 3.1e-3*aux
    gld = 0.4373e-3*dend
    gls = 1.44e-3*soma
    Eld=-97
    Ea=-35
    
    Vea, Vis, Ved, Vid = y.T
    
    Ii = gi * (Vid - Vis)
    Ia = -ga * (Vea + Ea)
    Ie = ge * (Vea - Ved)
    Ild = gld * (Eld - Vid + Ved)  #  current from inside to outside
    Ils = gls * (Vis - Els)  # current from outside to inside
    
    return Ii, Ia, Ie, Ild, Ils

def deriv_area(aux=1, dend=1, soma=1):
    def partial_func(y, t, f1, f2):
        return deriv(y, t, f1, f2,
          Cma=30e-3*aux, ga=3.1e-3*aux,
          Cmd=3.28e-3*dend, gld=0.4373e-3*dend,
          Cms=1.44e-3*soma, gls=1.44e-3*soma)
    return partial_func