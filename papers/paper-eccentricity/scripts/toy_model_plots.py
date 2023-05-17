import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata

from figure_config import fig_path, data_path

degree_format_str = '${x:.0f}\degree$'

def load_data(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def b_v_e_map(data):
    plt.figure()
    plt.pcolormesh(data['bs'], data['v0s'], data['e'], vmin=0, vmax=1, shading='gouraud')
    plt.colorbar().set_label('$e$')
    plt.xlabel('$b/\mathrm{pc}$')
    plt.ylabel('$v_0/\mathrm{km\,s^{-1}}$')

def b_v_th_map(data):
    plt.figure()
    plt.pcolormesh(data['bs'], data['v0s'], np.degrees(data['theta']), vmin=20, vmax=180, shading='gouraud')
    plt.colorbar().set_label(r'$\theta/\degree$')
    plt.xlabel('$b/\mathrm{pc}$')
    plt.ylabel('$v_0/\mathrm{km\,s^{-1}}$')

def th_v_e_map(data):
    th,v = np.meshgrid(np.linspace(0,np.pi, 200), data['v0s']) 
    _,v_data = np.meshgrid(data['bs'], data['v0s'])
    e = griddata((data['theta'].ravel(), v_data.ravel()), data['e'].ravel(), (th, v))

    plt.figure()
    plt.pcolormesh(np.degrees(th), v, e, vmin=0, vmax=1, shading='gouraud')
    plt.colorbar().set_label('$e$')
    plt.xlabel(r'$\theta$')
    plt.gca().xaxis.set_major_formatter(degree_format_str)
    plt.ylabel('$v_0/\mathrm{km\,s^{-1}}$')

def min_e_plot(data):
    min_e_ind = np.argmin(data['e'], axis=-1)
    fig, axes = plt.subplots(3,1, sharex='col')
    
    axes[0].plot(data['v0s'], data['e'][np.indices(min_e_ind.shape)[0],min_e_ind])
    axes[0].set_ylabel(r'$e_\mathrm{min}$')

    axes[1].plot(data['v0s'], np.degrees(data['theta'][np.indices(min_e_ind.shape)[0],min_e_ind]))
    axes[1].set_ylabel(r'$\theta_\mathrm{min}$')
    axes[1].yaxis.set_major_formatter(degree_format_str)

    axes[2].plot(data['v0s'], data['bs'][min_e_ind])
    axes[2].set_ylabel(r'$b_\mathrm{min}/\mathrm{pc}$')

    axes[2].set_xlabel('$v_0/\mathrm{km\,s^{-1}}$')

def th_e_curves(data):
    from matplotlib.collections import LineCollection

    fig = plt.figure()
    lines = LineCollection(list(map(np.column_stack, zip(np.degrees(data['theta']), data['e']))), array=data['v0s'])
    plt.gca().add_collection(lines)
    plt.xlim(0,180)
    cb = fig.colorbar(lines)
    cb.set_label('$v_0/\mathrm{km\,s^{-1}}$')

    plt.ylabel('$e$')
    plt.xlabel(r'$\theta$')
    plt.gca().xaxis.set_major_formatter(degree_format_str)
    
def plot_specific_th_e(data, v0s, simdata=None):
    plt.figure()
    for v0 in v0s:
        i = np.argmin(abs(np.array(data['v0s'])-v0))
        plt.plot(np.degrees(data['theta'][i]), data['e'][i], label=f"$v_0 = {data['v0s'][i]:3.0f}"r'/\mathrm{km\,s^{-1}}$')


    if simdata is not None:
        plot_sim_data_th_e(plt.gca(), simdata)
    
    plt.ylabel('$e$')
    plt.xlabel(r'$\theta$')
    plt.gca().xaxis.set_major_formatter(degree_format_str)
    plt.legend()

def plot_sim_data_th_e(ax, simdata):
    marker_kwargs = {"edgecolor":"k", "lw":0.5}
    for (l,s),m in zip(simdata.items(), 'ov^sd'):
        ax.scatter(s['thetas'], s['median_eccs'],
                   c=s['mass_res'], cmap='PuBu',
                   marker=m, **marker_kwargs, label=l,
                   zorder=3
                   )


def hr_well_fitting_models_plot(curve_shift=None):

    plt.figure()
    plt.ylabel('$e$')
    plt.xlabel(r'$\theta$')
    plt.gca().xaxis.set_major_formatter(degree_format_str)

    d = load_data(data_path('well_fitting_e_0.90_model_curve.pkl'))
    l, = plt.plot(np.degrees(d['theta'])+curve_shift, d['e'], color='tab:orange')
    plt.plot(np.degrees(d['theta']), d['e'], alpha=0.5, color=l.get_color())


    sim_data = load_data(data_path('deflection_angles_e0-0.900.pickle')) 
    plot_sim_data_th_e(plt.gca(), {'1':sim_data090})




sim_data090 = load_data(data_path('deflection_angles_e0-0.900.pickle')) 
sim_data099 = load_data(data_path('deflection_angles_e0-0.990.pickle')) 

#data = load_data(data_path('hernquist_b_v_scan_es_0.90_df_0.3.pkl'))
#plot_specific_th_e(data, [750,800,860], {'e=0.90': sim_data090, 'e=0.99':sim_data099})


# good fit
#data = load_data(data_path('old_scans/g05_b_v_scan_es_0.90_df_0.5.pkl'))
#plot_specific_th_e(data, [470], {'e=0.90': sim_data090})
#th_v_e_map(data)

#data = load_data(data_path('g05_b_v_scan_es_0.85_df_0.3.pkl'))
##data = load_data(data_path('g05_3_b_v_scan_es_0.90_df_0.3.pkl'))
#plot_specific_th_e(data, [470,500,560, 650], {'e=0.90': sim_data090,'e=0.99':sim_data099})
#th_v_e_map(data)

#b_v_e_map(data)
#b_v_th_map(data)
#th_v_e_map(data)
##min_e_plot(data)
#th_e_curves(data)

hr_well_fitting_models_plot(curve_shift=-12)

plt.show()
