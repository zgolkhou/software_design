"""
This script will try to recreate the amplitude-versus-energy and
amplitude-versus-duration panels from Hawley et al 2014 (ApJ 797, 12)
Figure 10
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mdwarf_utils import duration_from_energy
from mdwarf_utils import fwhm_from_duration
from mdwarf_utils import amplitude_from_fwhm_energy

from plot_utils import make_distribution_plot, make_density_plot

import numpy as np

dtype = np.dtype([('start_dex', float), ('stop_dex', float),
                  ('start', float), ('stop', float), ('peak', float),
                  ('rise', float), ('decay', float), ('amp', float),
                  ('e_dur', float), ('e_dur_rise', float),
                  ('e_dur_decay', float), ('flag', float),
                  ('ppl_flare', float), ('ppl_month', float),
                  ('components', float)])

control_data = np.genfromtxt('data/gj1243_master_flares.tbl', dtype=dtype)
valid = np.where(np.logical_not(np.isnan(control_data['amp'])))
control_data = control_data[valid]
valid = np.where(np.logical_and(control_data['amp']>0.0, control_data['e_dur']>0.0))
control_data = control_data[valid]

log_ekp_quiescent = 30.67*2.

control_log_ekp = log_ekp_quiescent + np.log10(control_data['e_dur'])
duration = 24.0*30.0*(control_data['stop']-control_data['start'])

log_eu = control_log_ekp + np.log10(0.65)

eu = np.power(10.0, log_eu)
rng = np.random.RandomState(88)

t_fwhm = fwhm_from_duration(duration)
amplitude_u = amplitude_from_fwhm_energy(t_fwhm, eu)

# Convert to Kepler amplitude.
# See paragraph before section 3 of Hawley et al 2014
amplitude_kp = amplitude_u/0.65 

amp_rel = amplitude_kp/np.power(10.0, log_ekp_quiescent)

def _build_activity_model():
    """
    Read in data taken from West et al. 2008 (AJ 135, 785).
    For each spectral type, return the parametrs A, tau, B
    needed to model spectral activity as a function of
    distance from the Galactic Plane as
    fraction active = A*exp(-z/tau) + B
    """
    data_dir = 'data/activity_by_type'

    dtype = np.dtype([('z', float), ('frac', float),
                      ('min', float), ('max', float)])

    model_aa = []
    model_bb = []
    model_tau = []
    for i_type in range(9):
        model_type = 'M%d' % i_type
        data_name = os.path.join(data_dir, '%s.txt' % model_type)
        data = np.genfromtxt(data_name, dtype=dtype)

        sigma_arr = []
        for nn, xx in zip(data['min'], data['max']):
            if nn>1.0e-20 and xx<0.999:
                sigma = 0.5*(xx-nn)
            else:
                sigma = xx-nn
            sigma_arr.append(sigma)
        aa, tau, bb = fit_to_exp_decay(data['z'], data['frac'], np.array(sigma_arr))
        model_aa.append(aa)
        model_bb.append(bb)
        model_tau.append(tau)

    return np.array(model_aa), np.array(model_bb), np.array(model_tau)
  
dx=0.05
plt.figsize = (30, 30)
plt.subplot(2,2,1)
make_density_plot(control_log_ekp, np.log10(amp_rel), dx, cmax=80, dc=20)
plt.xlabel('Log(E_kp)')
plt.ylabel('Log(simulated amplitude)')
plt.xlim(27, 34)
plt.ylim(-5, 1)

plt.subplot(2,2,2)
make_density_plot(control_log_ekp, np.log10(control_data['amp']), dx,
                  cmax=80, dc=20)
plt.xlabel('Log(E_kp)')
plt.ylabel('Log(amplitude')
plt.xlim(27, 34)
plt.ylim(-5, 1)

plt.subplot(2,2,3)
hh, = make_distribution_plot(np.log10(amp_rel), dx, color='k')
header_list = [hh]
label_list = ['simulation']
hh, = make_distribution_plot(np.log10(control_data['amp']), dx, color='r')
header_list.append(hh)
label_list.append('data')
plt.xlabel('Log(amplitude)')
plt.ylabel('N')
plt.legend(header_list, label_list, loc=0, fontsize=10)


