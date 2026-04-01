import sys
sys.path.append('/home/joaopedrolopez/Downloads/AvaliacaoExperimental/Experimentos/novos/Detecting_Paleoclimate_Transitions_with_LERM/notebooks/Main_Analysis')
from speedupy.speedupy import maybe_deterministic
import pyleoclim as pyleo
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import ammonyte as amt
from tqdm import tqdm
from speedupy.speedupy import initialize_speedupy, deterministic

@maybe_deterministic
def cell_4(group_names):
    series_list = []
    color_list = sns.color_palette('colorblind')
    for name in group_names:
        with open('../data/LR04cores_spec_corr/' + name[-3:] + '_LR04age.txt', 'rb') as handle:
            lines = handle.readlines()
            time = []
            d18O = []
            for x in lines:
                t0 = x.decode()
                t1 = t0.split()
                line_time = float(format(float(t1[1]), '10f'))
                t2 = x.decode()
                t3 = t2.split()
                line_d18O = float(format(float(t3[2]), '10f'))
                if line_time <= 4000:
                    time.append(line_time)
                    d18O.append(line_d18O)
            series = pyleo.Series(value=d18O, time=time, label=name, time_name='Yr', time_unit='ka', value_name='$\\delta^{18}O$', value_unit='permil')
        series_list.append(series)
    max_time = min([max(series.time) for series in series_list])
    min_time = max([min(series.time) for series in series_list])
    t12 = []
    for series in series_list:
        t4 = series.slice((min_time, max_time))
        t12.append(t4.interp())
    ms = pyleo.MultipleSeries(t12)
    fig, ax = ms.stackplot(colors=color_list[:len(ms.series_list)], figsize=(8, 10))
    return [color_list, ms]

@maybe_deterministic
def detect_transitions(series, transition_interval=None):
    """Function to detect transitions across a confidence interval
    
    Parameters
    ----------
    
    series : pyleo.Series, amt.Series
        Series to detect transitions upon
        
    transition_interval : list,tuple
        Upper and lower bound for the transition interval
    
    Returns
    -------
    
    transitions : list
        Timing of the transitions of the series across its confidence interval
    """
    series_fine = series.interp(step=1)
    if transition_interval is None:
        upper, lower = amt.utils.sampling.confidence_interval(series)
    else:
        upper, lower = transition_interval
    above_thresh = np.where(series_fine.value > upper, 1, 0)
    below_thresh = np.where(series_fine.value < lower, 1, 0)
    transition_above = np.diff(above_thresh)
    transition_below = np.diff(below_thresh)
    upper_trans = series_fine.time[1:][np.diff(above_thresh) != 0]
    lower_trans = series_fine.time[1:][np.diff(below_thresh) != 0]
    full_trans = np.zeros(len(transition_above))
    last_above = 0
    last_below = 0
    for i in range(len(transition_above)):
        above = transition_above[i]
        below = transition_below[i]
        if above != 0:
            if last_below + above == 0:
                loc = int((i + below_pointer) / 2)
                full_trans[loc] = 1
                last_below = 0
            last_above = above
            above_pointer = i
        if below != 0:
            if last_above + below == 0:
                loc = int((i + above_pointer) / 2)
                full_trans[loc] = 1
                last_above = 0
            last_below = below
            below_pointer = i
    transitions = series_fine.time[1:][full_trans != 0]
    return transitions

@deterministic
def cell_6(ms, func_globals=None):
    lp_rm = {}
    lp_fi = {}
    m = 13
    t5 = ms.common_time()
    for idx, series in enumerate(t5.series_list):
        t6 = series.convert_time_unit('Years')
        t7 = t6.interp()
        series = t7.detrend(method='savitzky-golay')
        amt_series = amt.Series(time=series.time, value=series.value, time_name=series.time_name, value_name=series.value_name, time_unit=series.time_unit, value_unit=series.value_unit, label=series.label, clean_ts=False, sort_ts=None)
        td = amt_series.embed(m=m)
        print(f'{series.label} tau is : {td.tau}')
        eps = td.find_epsilon(eps=1, target_density=0.05, tolerance=0.01)
        rm = eps['Output']
        lp_series = rm.laplacian_eigenmaps(w_size=50, w_incre=5)
        lp_series = lp_series.convert_time_unit('ka')
        lp_fi[series.label] = lp_series
    return (ms, lp_fi, lp_series)

@maybe_deterministic
def cell_7(ms, group_names, lp_fi, lp_series, color_list):
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 25
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    fig, axes = plt.subplots(nrows=len(group_names), ncols=1, sharex=True, figsize=(16, 10))
    transition_timing = []
    for idx, site in enumerate(group_names):
        ts = lp_fi[site]
        ts.label = lp_series.label
        ts.value_name = 'FI'
        ts.value_unit = None
        ts.time_name = 'Yr'
        ts.time_unit = 'ka'
        ax = axes[idx]
        ts_smooth = amt.utils.fisher.smooth_series(series=ts, block_size=3)
        upper, lower = amt.utils.sampling.confidence_interval(series=ts, upper=95, lower=5, w=50, n_samples=10000)
        transitions = detect_transitions(ts_smooth, transition_interval=(upper, lower))
        transition_timing.append(transitions[0])
        ts.confidence_smooth_plot(ax=ax, background_series=ms.series_list[idx], transition_interval=(upper, lower), block_size=3, color=color_list[idx], figsize=(12, 6), legend=True, lgd_kwargs={'loc': 'upper left'}, hline_kwargs={'label': None}, background_kwargs={'ylabel': 'delta^{18}O$ [permil]', 'legend': False, 'linewidth': 0.8, 'color': 'grey', 'alpha': 0.8})
        ax.axvline(transition_timing[idx], color='grey', linestyle='dashed', alpha=0.5)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(x=-0.08, y=2.5, s=site, horizontalalignment='right', transform=trans, color=color_list[idx], weight='bold', fontsize=20)
        t9 = ax.spines['left']
        t9.set_visible(True)
        t10 = ax.spines['right']
        t10.set_visible(False)
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()
        t8 = ax.get_legend()
        t8.remove()
        ax.set_title(None)
        ax.grid(visible=False, axis='y')
        if idx != len(group_names) - 1:
            ax.set_xlabel(None)
            t11 = ax.spines[['bottom']]
            t11.set_visible(False)
            ax.tick_params(bottom=False)
        ax.xaxis.label.set_fontsize(25)
        ax.yaxis.label.set_fontsize(25)
        ax.set_yticks(ticks=np.array([0, 5]))
    return transition_timing

@maybe_deterministic
def cell_8(transition_timing):
    np.mean(transition_timing)

@maybe_deterministic
def cell_9(transition_timing):
    np.std(transition_timing)

@initialize_speedupy
def main():
    import warnings
    warnings.filterwarnings('ignore')
    group_names = ['ODP 925', 'ODP 927', 'ODP 929', 'ODP 846', 'ODP 849']
    color_list, ms = cell_4(group_names)
    ms, lp_fi, lp_series = cell_6(ms, func_globals=globals())
    transition_timing = cell_7(ms, group_names, lp_fi, lp_series, color_list)
    cell_8(transition_timing)
    cell_9(transition_timing)

main()
