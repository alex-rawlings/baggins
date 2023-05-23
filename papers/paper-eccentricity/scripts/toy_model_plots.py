import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata, interp1d

from figure_config import fig_path, data_path
from Plotter import Plotter

degree_format_str = '{x:.0f}°'

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
    
def plot_specific_th_e(data, v0s, simdata=None, shift=0):
    plt.figure()
    for v0 in v0s:
        i = np.argmin(abs(np.array(data['v0s'])-v0))
        plt.plot(np.degrees(data['theta'][i]) + shift, data['e'][i], label=f"$v_0 = {data['v0s'][i]:3.0f}"r'/\mathrm{km\,s^{-1}}$')


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


def paper_plots():
    plotter = Plotter()

    fig, axes = plt.subplots(1,2,sharey='row')
    fig.set_figwidth(2*fig.get_figwidth())

    axes[0].set_ylabel('$e$')
    axes[0].set_ylim(0,1)
    for ax in axes:
        ax.set_xlabel(r'$\theta$')
        ax.xaxis.set_major_formatter(degree_format_str)
        ax.set_xlim(20,170)
        ax.set_xticks(np.arange(30,180,30))

    data = load_data(data_path('well_fitting_e_0.90_model_curve.pkl'))
    curve_shift = -12
    for e_s, d in zip(data['e_spheroids'], data['res']):
        if e_s <0.9: continue
        a = 1 if e_s == 0.905 else 0.5
        l, = axes[0].plot(np.degrees(d['theta'])+curve_shift, d['e'], color='tab:blue', alpha=a)
    #axb = axes[0].secondary_xaxis('top', 
    #            functions=(interp1d(np.degrees(d['theta']), d['b']*1e3,
    #                                bounds_error=False, fill_value='extrapolate'),
    #                       interp1d(d['b']*1e3, np.degrees(d['theta']),
    #                                bounds_error=False, fill_value='extrapolate')))
    #axb.set_xlabel('$b/\mathrm{pc}$')
    #axb.set_xticks([1,2,3,5,10,20])
    #axes[0].tick_params(top=False)
    print("b90 for e_0=0.90", np.interp(90,np.degrees(d['theta'][::-1]),d['b'][::-1]*1e3))


    axes[0].arrow(35-curve_shift,0.1,curve_shift,0, length_includes_head=True,
                  width=0.01, head_width=0.02, head_length=3, edgecolor='none',
                  facecolor='k')
    axes[0].text(22,0.13,"Model shift")

    axes[0].text(22,0.95,"$e_0=0.90$")

    sim_data = load_data(data_path('deflection_angles_e0-0.900.pickle')) 
    for k, g in itertools.groupby(sorted(zip(sim_data['mass_res'], sim_data['thetas'], sim_data['median_eccs'])),
                                  lambda x: x[0]):
        th, e = np.array(list(g)).T[1:]
        if not np.any(np.isfinite(e)):
            continue #some nan values in this dataset
        plotter.scatter(th,e, label=rf"{k:.0f}", zorder=2, ax=axes[0])


    data = load_data(data_path('well_fitting_e_0.99_model_curve.pkl'))
    curve_shift = -6
    for e_s, d in zip(data['e_spheroids'], data['res']):
        if e_s < 0.9: continue
        a = 1 if e_s == 0.905 else 0.5
        #a = 0.7
        axes[1].plot(np.degrees(d['theta'])+curve_shift, d['e'], alpha=a, color='tab:blue')


    #axb = axes[1].secondary_xaxis('top', 
    #            functions=(interp1d(np.degrees(d['theta']), d['b']*1e3,
    #                                bounds_error=False, fill_value='extrapolate'),
    #                       interp1d(d['b']*1e3, np.degrees(d['theta']),
    #                                bounds_error=False, fill_value='extrapolate')))
    #axb.set_xlabel('$b/\mathrm{pc}$')
    #axb.set_xticks([1,2,3,5,10,20])
    #axes[1].tick_params(top=False)
    print("b90 for e_0=0.99", np.interp(90,np.degrees(d['theta'])[::-1],d['b'][::-1]*1e3))

    axes[1].arrow(30-curve_shift,0.1,curve_shift,0, length_includes_head=True,
                  width=0.01, head_width=0.02, head_length=3, edgecolor='none',
                  facecolor='k')

    axes[1].text(22,0.95,"$e_0=0.99$")

    sim_data = load_data(data_path('deflection_angles_e0-0.990.pickle')) 
    for k, g in itertools.groupby(sorted(zip(sim_data['mass_res'], sim_data['thetas'], sim_data['median_eccs'])),
                                  lambda x: x[0]):
        th, e = np.array(list(g)).T[1:]
        plotter.scatter(th,e, label=f"{k:.0f}", zorder=2, ax=axes[1])

    axes[1].legend(ncol=1, title=r'$M_\bullet/m_\star$', loc='lower right')
    plt.savefig(fig_path('theta_e_sim_and_model.pdf'))


#sim_data090 = load_data(data_path('deflection_angles_e0-0.900.pickle')) 
#sim_data099 = load_data(data_path('deflection_angles_e0-0.990.pickle')) 

#data = load_data(data_path('hernquist_b_v_scan_es_0.90_df_0.3.pkl'))
#plot_specific_th_e(data, [750,800,860], {'e=0.90': sim_data090, 'e=0.99':sim_data099})

#data = load_data(data_path('g05_3_b_v_scan_es_0.90_df_0.5.pkl'))
###data = load_data(data_path('g05_3_b_v_scan_es_0.90_df_0.3.pkl'))
#plot_specific_th_e(data, [450, 560,590], {'e=0.90': sim_data090,'e=0.99':sim_data099}, shift=0)
#th_v_e_map(data)

#b_v_e_map(data)
#b_v_th_map(data)
#th_v_e_map(data)
##min_e_plot(data)
#th_e_curves(data)


paper_plots()

plt.show()
