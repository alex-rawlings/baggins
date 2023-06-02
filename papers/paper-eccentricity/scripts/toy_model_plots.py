import itertools
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata, interp1d

from figure_config import fig_path, data_path, marker_cycle, color_cycle, color_cycle_shuffled, marker_kwargs

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
    for (l,s),m in zip(simdata.items(), 'ov^sd'):
        ax.scatter(s['thetas'], s['median_eccs'],
                   c=s['mass_res'], cmap='PuBu',
                   marker=m, **marker_kwargs, label=l,
                   zorder=3
                   )


def paper_th_e_curve_plot():

    fig, axdict = plt.subplot_mosaic(
                                """
                                .AAABBB.
                                CDDDEEEF
                                CDDDEEEF
                                CDDDEEEF
                                """,
                                figsize=(6,4)
    )
    # share the axis limits
    # marginal axes
    axdict["A"].sharex(axdict["D"])
    axdict["B"].sharex(axdict["D"]) # E shared with D later, makes ticks work
    axdict["C"].sharey(axdict["D"])
    axdict["F"].sharey(axdict["E"])
    # main panels
    axdict["D"].sharex(axdict["E"])
    axdict["D"].sharey(axdict["E"])

    # turn off some ticks
    for k in "AB":
        axdict[k].set_xticklabels([])
        axdict[k].set_yticklabels([])
        axdict[k].set_yticks([])
    for k in "DEF":
        axdict[k].set_yticklabels([])
    for k in "CF":
        axdict[k].set_xticklabels([])
        axdict[k].set_xticks([])

    axdict["C"].set_ylabel('$e$')
    axdict["C"].set_ylim(0,1)
    for k in "DE":
        axdict[k].set_xlabel(r'$\theta_\mathrm{defl}$')
        axdict[k].xaxis.set_major_formatter(degree_format_str)
    for k in "ABDE":
        axdict[k].set_xlim(20,170)
        axdict[k].set_xticks(np.arange(30,180,30))

    data = load_data(data_path('well_fitting_e_0.90_model_curve.pkl'))
    curve_shift = -12

    color = None
    for e_s, d in zip(data['e_spheroids'], data['res']):
        if e_s <0.9: continue
        a = 1 if e_s == 0.905 else 0.5
        l, = axdict["D"].plot(np.degrees(d['theta'])+curve_shift, d['e'], color=color, alpha=a)
        color = l.get_color()
    print("b90 for e_0=0.90", np.interp(90,np.degrees(d['theta'][::-1]),d['b'][::-1]*1e3))


    axdict["D"].arrow(35-curve_shift,0.1,curve_shift,0, length_includes_head=True,
                  width=0.01, head_width=0.02, head_length=3, edgecolor='none',
                  facecolor='k')
    axdict["D"].text(22,0.13,"Model shift")

    axdict["A"].set_title("$e_0=0.90$")

    data = load_data(data_path('well_fitting_e_0.99_model_curve.pkl'))
    curve_shift = -6

    color=None
    for e_s, d in zip(data['e_spheroids'], data['res']):
        if e_s < 0.9: continue
        a = 1 if e_s == 0.905 else 0.5
        l, = axdict["E"].plot(np.degrees(d['theta'])+curve_shift, d['e'], alpha=a, color=color)
        color = l.get_color()



    print("b90 for e_0=0.99", np.interp(90,np.degrees(d['theta'])[::-1],d['b'][::-1]*1e3))

    axdict["E"].arrow(30-curve_shift,0.1,curve_shift,0, length_includes_head=True,
                  width=0.01, head_width=0.02, head_length=3, edgecolor='none',
                  facecolor='k')

    axdict["B"].set_title("$e_0=0.99$")

    def plot_sim_data(sim_data, ax, axmt, axme):
        ax.set_prop_cycle(color_cycle_shuffled)
        tbins = np.arange(*ax.get_xlim(), 5)
        ebins = np.arange(*ax.get_ylim(), 0.1)
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
            # add KDE to marginal
            kde_t = scipy.stats.gaussian_kde(th[~np.isnan(th)])
            t_pts = np.linspace(*ax.get_xlim(), 1000)
            axmt.plot(t_pts, kde_t(t_pts))
            kde_e = scipy.stats.gaussian_kde(e[~np.isnan(e)])
            e_pts = np.linspace(*ax.get_ylim(), 1000)
            axme.plot(kde_e(e_pts), e_pts)
    axdict["C"].invert_xaxis()

    plot_sim_data(load_data(data_path('deflection_angles_e0-0.900.pickle')), axdict["D"], axdict["A"], axdict["C"])

    plot_sim_data(load_data(data_path('deflection_angles_e0-0.990.pickle')), axdict["E"], axdict["B"], axdict["F"])

    axdict["E"].legend(ncol=1, title=r'$M_\bullet/m_\star$', loc='lower right')
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

    e_ax = axdict['C']
    e_ax.set_xlim((0,6))
    e_ax.set_ylim((0,1))
    e_ax.set_xlabel('$t/\mathrm{Myr}$')
    e_ax.set_ylabel(r'$e$')
    e_ax.set_yscale('eccentricity')

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
    e_ax.set_prop_cycle(color_cycle)
    e_ax.axvline(tmax, color='silver')
    e_ax.axvline(tmax, color='silver')
    for e_s, ax_orbit_key in zip([0.2,0.9], 'AB'):
        ax_orbit = axdict[ax_orbit_key]
        plot_potential(e_s, ax_orbit)
        plot_dset(load_data(data_path(f'sample_orbits_e_s_{e_s:.1f}.pkl')), ax_orbit, e_ax, tmax=tmax) 
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
