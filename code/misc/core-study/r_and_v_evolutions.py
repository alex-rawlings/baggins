import numpy as np
import matplotlib.pyplot as plt


def plot_r_and_v_evolutions(v_kick):
    
    gad_t_fac = 0.978
    data_file = '/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data/kick-vel-' + v_kick + '.txt'
    data = np.loadtxt(data_file, skiprows = 1)
    t = data[:,0]*gad_t_fac
    r = data[:,1]
    v = data[:,2]
    r_mBH = data[:,4]

    line_text = '$r_\mathrm{b,0}$'
    #radius of stellar core scoured by the SMBH binary
    rb = 0.58
    sigma_1D = 270.
    sigma_text = '$\sigma_\star(r<r_\mathrm{b,0})$'

    fig, ax = plt.subplots(2,1)
    end_i=-1
    for i in range(2,len(t)-1):
        t0 = t[i]
        v_used = v[(t-t0<0.1) & (t>=t0)]
        r_used = r[(t-t0<0.1) & (t>=t0)]
        if not any(r_used > 0.1):
            if (np.median(v_used) < 25):
                inds = v_used<25
                ind = np.argmax(inds)
                #print(v_kick,  i+ind)
                end_i = i+ind
                break
    t_end = t[end_i]        
    if end_i > 0:
        ax[0].axvline(t_end, color='k', label='$t_\mathrm{settle}$')
        ax[1].axvline(t_end, color='k', label='$t_\mathrm{settle}$')
    else:
        print("No settling found for kick velocity ", v_kick)  


    ax[0].axhline(sigma_1D, color='k', ls='--', lw=1, label=sigma_text)
    ax[1].axhline(rb, color='k', ls='--', lw=1, label=line_text)
    
    ax[1].semilogy(t, r, '.')
    ax[0].semilogy(t,v, '.')

    ax[0].set_ylabel('$v/\mathrm{kms}^{-1}$')
    ax[1].set_ylabel('$r/\mathrm{kpc}$')
    ax[1].set_xlabel('$t/\mathrm{Gyr}$')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('time-evolution-'+ v_kick+'.pdf')
    plt.close(fig)


v_list = ['0000', '0060','0120', '0180',  '0240',   '0300' ,  '0360' ,  '0420' ,  '0480' ,
  '0540' ,  '0600' ,  '0660' ,  '0720' ,  '0780' ,  '0840' ,  '0900' ,  '0960' ,  '1020' ,  '1080' ,
  '1140' ,  '1200' ,  '1260' ,  '1320' ,  '1380' ,  '1440' ,'1500' ,  '1560',   '1620' ,  '1680' ,  
  '1740' ,  '1800' ,  '2000']

for v in v_list:
    print(v)
    plot_r_and_v_evolutions(v)