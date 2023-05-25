import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata, interp1d

from figure_config import fig_path, data_path, marker_cycle, color_cycle

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


def paper_th_e_curve_plot():

    fig, axes = plt.subplots(1,2,sharey='row', figsize=(6,3.))

    axes[0].set_ylabel('$e$')
    axes[0].set_ylim(0,1)
    for ax in axes:
        ax.set_xlabel(r'$\theta_\mathrm{defl}$')
        ax.xaxis.set_major_formatter(degree_format_str)
        ax.set_xlim(20,170)
        ax.set_xticks(np.arange(30,180,30))

    data = load_data(data_path('well_fitting_e_0.90_model_curve.pkl'))
    curve_shift = -12
    color = next(iter(color_cycle[-1:]))['color'] 

    for e_s, d in zip(data['e_spheroids'], data['res']):
        if e_s <0.9: continue
        a = 1 if e_s == 0.905 else 0.5
        l, = axes[0].plot(np.degrees(d['theta'])+curve_shift, d['e'], color=color, alpha=a)
    print("b90 for e_0=0.90", np.interp(90,np.degrees(d['theta'][::-1]),d['b'][::-1]*1e3))


    axes[0].arrow(35-curve_shift,0.1,curve_shift,0, length_includes_head=True,
                  width=0.01, head_width=0.02, head_length=3, edgecolor='none',
                  facecolor='k')
    axes[0].text(22,0.13,"Model shift")

    axes[0].set_title("$e_0=0.90$")

    data = load_data(data_path('well_fitting_e_0.99_model_curve.pkl'))
    curve_shift = -6
    for e_s, d in zip(data['e_spheroids'], data['res']):
        if e_s < 0.9: continue
        a = 1 if e_s == 0.905 else 0.5
        axes[1].plot(np.degrees(d['theta'])+curve_shift, d['e'], alpha=a, color=color)


    print("b90 for e_0=0.99", np.interp(90,np.degrees(d['theta'])[::-1],d['b'][::-1]*1e3))

    axes[1].arrow(30-curve_shift,0.1,curve_shift,0, length_includes_head=True,
                  width=0.01, head_width=0.02, head_length=3, edgecolor='none',
                  facecolor='k')

    axes[1].set_title("$e_0=0.99$")

    def plot_sim_data(sim_data, ax):
        for (k, g), m in zip(
                            itertools.groupby(
                            sorted(zip(sim_data['mass_res'],
                                        sim_data['thetas'],
                                        sim_data['median_eccs'])),
                                      lambda x: x[0]),
                            marker_cycle):
            th, e = np.array(list(g)).T[1:]
            if not np.any(np.isfinite(e)):
                continue #some nan values in this dataset
            ax.plot(th,e, label=rf"{k:.0f}", zorder=2, ls='none', **m)

    plot_sim_data(load_data(data_path('deflection_angles_e0-0.900.pickle')), axes[0])

    plot_sim_data(load_data(data_path('deflection_angles_e0-0.990.pickle')), axes[1])

    axes[1].legend(ncol=1, title=r'$M_\bullet/m_\star$', loc='lower right')
    fig.savefig(fig_path('theta_e_sim_and_model.pdf'))

def paper_orbit_plot():
    fig, axdict = plt.subplot_mosaic(
                        """
                        AABB
                        AABB
                        CCCC
                        """,
                        figsize=(6,5)
                        )
    for k in 'AB':
        axdict[k].set_aspect('equal')
        axdict[k].set_xlim((-80,80))
        axdict[k].set_ylim((-120,40))
        axdict[k].set_xlabel('$x/\mathrm{pc}$')
        axdict[k].set_ylabel('$y/\mathrm{pc}$')

    axdict['C'].set_xlim((0,6))
    axdict['C'].set_ylim((0,1))
    axdict['C'].set_xlabel('$t/\mathrm{Myr}$')
    axdict['C'].set_ylabel(r'$e$')
    axdict['C'].set_yscale('eccentricity')

    def plot_potential(e_spheroid, ax):
        e2s = e_spheroid**2
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
        A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)

        X,Y = np.meshgrid(np.linspace(*ax.get_xlim(), 200), np.linspace(*ax.get_ylim(), 200))
        pot = A3 * X**2 + A1*Y**2 # constant magnitude factors aren't needed here

        ax.contour(X,Y,pot, colors='silver', levels=10, linewidths=1., zorder=0, linestyles='--')
    
    def plot_dset(dset, orbit_ax, e_ax, tmax=None):
        for d in dset:
            t = d['t']*1e3
            if tmax is None:
                tmax = np.max(t)
            l, = e_ax.plot(t, d['e'])
            orbit_ax.plot(*d['x'][:,t<=tmax]*1e3, color=l.get_color(), lw=1.5)

    tmax = 2.5
    axdict['C'].axvline(tmax, color='silver')
    for e_s, ax_orbit_key in zip([0.2,0.9], 'AB'):
        ax_orbit = axdict[ax_orbit_key]
        plot_potential(e_s, ax_orbit)
        plot_dset(load_data(data_path(f'sample_orbits_e_s_{e_s:.1f}.pkl')), ax_orbit, axdict['C'], tmax=tmax) 
        ax_orbit.set_title(fr'$e_\mathrm{{s}}={e_s:.1f}$')


    fig.savefig(fig_path('sample_model_orbits.pdf'))

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


paper_th_e_curve_plot()
paper_orbit_plot()

#plt.show()
