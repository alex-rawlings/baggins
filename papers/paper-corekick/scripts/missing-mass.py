import numpy as np
import figure_config
import pickle
from scipy import integrate
import matplotlib.pyplot as plt


col_list = figure_config.color_cycle_shuffled.by_key()["color"]

def sersic_fit(r, Re,n):
    '''

    :param r:
    :param Re: effective radius
    :param n: Sérsic index
    :return:
    '''
    b = 2.0 * n - 0.33333333 + 0.009876 * (1/n)
    mu = np.exp(-b*((r/Re)**(1/n)-1))
    return mu


def core_sersic_fit(r, rb,Re, n, mub, g, a):
    '''

    :param r:
    :param rb: break radius
    :param Re: effective radius
    :param n: Sérsic index
    :param mub: normalisation factor Sigma_b
    :param g: gamma, inner profile slope index
    :param a: alpha, transition index
    :return:
    '''

    # Based on work in Nasim et al. 2021, Graham et al. 2003, and Trujillo et al. 2004.

    b = 2.0 * n - 0.33333333 + 0.009876 * (1 / n)
    mudot = mub * 2**(-g/a)*np.exp(b*2**(1/(a*n))*(rb/Re)**(1/n)) #the mistake was here
    mu = mudot*(1+(rb/r)**a)**(g/a)*np.exp(-b*((r**a+rb**a)/Re**a)**(1/(n*a)))
    return mu

def mass_deficit(r, rb, re, n, log10densb, g, a):
    '''

    :param r:
    :param rb: break radius
    :param re: effective radius
    :param n: Sérsic index
    :param log10densb: normalisation factor Sigma_b
    :param g: gamma, inner profile slope index
    :param a: alpha, transition index
    :return: The integrand of the mass deficit equation
    '''
    mub = 10**log10densb

    # As we don't have Ie, we need to find the correct shift for the sersic fit based on the core-sersic
    # Calculating mu_cs and mu_c at the break radius
    mu_cs_rb = core_sersic_fit(rb, rb, re, n, mub, g, a)
    mu_s_rb = sersic_fit(rb, re, n)
    shift = np.log10(mu_cs_rb)-np.log10(mu_s_rb)
    return (10**shift *sersic_fit(r, re, n) - core_sersic_fit(r,rb, re, n, mub, g, a))* r


def missing_mass_plot(filename):
    '''

    :param filename: The path and dictionary file of the data. Dictionary should include break radius,
    effective radius, sersic index, normalistion factor sigma_b, inner profile slope index and the transition index
    :return:
    '''

    with open(filename, "rb") as f:
        data = pickle.load(f)

    # Kick velocities
    vkicks = data['rb'].keys()

    # An empty dictionry to save the data
    mdef_dict = dict()
    all_mdefs = np.zeros(len(vkicks), dtype=object)

    # Defining the random seed to get reproducible results
    rng = np.random.default_rng(seed=42)
    # Looping over all kick velocities
    for j, v in enumerate(vkicks):
        rb = data['rb'][v].flatten()      # break radius
        Re = data['Re'][v].flatten()         # effective radius
        n = data['n'][v].flatten()          # sersic index
        log10densb = data['log10densb'][v].flatten()# normalisation factor Sigma_b
        g = data['g'][v].flatten()         # gamma, inner profile slope index
        a = data['a'][v].flatten()

        nro_iter = 10000
        mdf_vkick = np.zeros(nro_iter)
        mdef_dict[v] = np.zeros(nro_iter)

        for i in range(nro_iter):
            rb_new, Re_new, n_new, log10densb_new, g_new, a_new = rng.choice(rb), rng.choice(Re), rng.choice(n), rng.choice(log10densb), rng.choice(g), rng.choice(a)
            print(rb_new,Re_new, n_new)

            print()
            print("THIS:")
            print(log10densb_new)
            print(g_new,a_new)
            print()
            return
            m, abserr = integrate.quad(mass_deficit,1e-3, rb_new, args=(rb_new, Re_new, n_new, log10densb_new, g_new, a_new))
            if np.isnan(m)==True:
                rb_new, Re_new, n_new, log10densb_new, g_new, a_new = rng.choice(rb), rng.choice(Re), rng.choice(
                    n), rng.choice(log10densb), rng.choice(g), rng.choice(a)

                m, abserr = integrate.quad(mass_deficit, 0, rb_new,
                                           args=(rb_new, Re_new, n_new, log10densb_new, g_new, a_new))
            mdef_dict[v][i] = 2*np.pi*3.5*m
            mdf_vkick[i] = 2*np.pi*3.5*m
            # There is one nan -value in the 820 case

        all_mdefs[j] = mdf_vkick
        print('done ' +v)
        #print(mdf_vkick)
        print()

    # Creating the figure
    fig, ax = plt.subplots()
    velocities = [int(key) for key in data['rb'].keys()]
    norm_val = np.median(mdef_dict['0000'])
    bp = ax.boxplot(all_mdefs/norm_val, positions=velocities, showfliers=False,
        whis=0, widths=50, manage_ticks=False, patch_artist=True, showcaps=False)

    for p in bp["boxes"]:
        p.set_facecolor(col_list[0])
        p.set_edgecolor(p.get_facecolor())
        p.set_alpha(0.3)
    for m in bp["medians"]:
        m.set_color("#003A74")
        m.set_linewidth(2)
        m.set_alpha(1)
    for w in bp["whiskers"]:
        w.set_alpha(0)

    ax.tick_params(axis="y", which="both", right=False)
    ax2 = ax.secondary_yaxis(
        "right", functions=(lambda x: x * norm_val, lambda x: x / norm_val)
    )
#    ax2.yaxis.set_major_formatter(r'$\mathrm{10**9}$')
    ax2.ticklabel_format(style='sci',useMathText=True)
    # Ax labels
    ax.set_ylabel(r"$M_\mathrm{def} / M_\mathrm{def,0}$")
    ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{kms}^{-1}$")
    ax2.set_ylabel(r"$M_\mathrm{def}/ \mathrm{M}_\odot$")
    # Saving the figure
    plt.show()
    #splt.savefig(figure_config.fig_path("missing-mass.pdf"))

data_file = '/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle'
missing_mass_plot(data_file)
