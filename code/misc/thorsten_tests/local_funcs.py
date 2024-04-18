import numpy as np
import scipy.interpolate
import baggins as bgs


__all__ = ["HMQ_generator", "get_hard_timespan"]

def HMQ_generator(dirs, merged=False, kick="None"):
    count = -1
    for d in dirs:
        HMQ_files = bgs.utils.get_files_in_dir(d)
        for f in HMQ_files:
            count += 1
            hmq = bgs.analysis.HMQuantitiesBinaryData.load_from_file(f)
            if merged:
                # highlight mergers
                if hmq.merger_remnant["merged"]:
                    # for mergers
                    if kick == "None":
                        # don't care about kick
                        alpha = 1
                    elif kick == "High":
                        # show mergers with high kicks
                        alpha = 1 if hmq.merger_remnant["kick"] > 1000 else 0.1
                    else:
                        # show mergers with low kicks
                        alpha = 1 if hmq.merger_remnant["kick"] < 1000 else 0.1
                else:
                    alpha = 0.1
            else:
                alpha = 1
            yield hmq, alpha, count, f


def get_hard_timespan(t, a, t_s, ah_s):
    f = scipy.interpolate.interp1d(t_s, ah_s, bounds_error=False, fill_value=(ah_s[0], ah_s[-1]))
    return np.sum(a < f(t)) * (t[1]-t[0])
