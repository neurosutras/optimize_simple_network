from nested.utils import *
from scipy.signal import butter, sosfiltfilt, sosfreqz, hilbert, periodogram, savgol_filter
from scipy.ndimage import gaussian_filter1d
from collections import namedtuple, defaultdict
from numbers import Number

mpl.rcParams['font.size'] = 16.
mpl.rcParams['font.sans-serif'] = 'Arial'


def get_gaussian_rate(duration, peak_loc, sigma, min_rate, max_rate, dt, wrap_around=True, buffer=0., equilibrate=0.):
    """

    :param duration: float
    :param peak_loc: float
    :param sigma: float
    :param min_rate: float
    :param max_rate: float
    :param dt: float
    :param wrap_around: bool
    :param buffer: float
    :param equilibrate: float
    :return: array
    """
    if wrap_around:
        t = np.arange(0., duration + dt / 2., dt)
        extended_t = np.concatenate([t - duration, t, t + duration])
        rate = (max_rate - min_rate) * np.exp(-((extended_t - peak_loc) / sigma) ** 2.) + min_rate
        before = np.array(rate[:len(t)])
        after = np.array(rate[2 * len(t):])
        within = np.array(rate[len(t):2 * len(t)])
        rate = within[:len(t)] + before[:len(t)] + after[:len(t)]
        buffer_len = int(buffer / dt)
        equilibrate_len = int(equilibrate / dt)
        if buffer_len > 0:
            rate = np.concatenate([rate[-buffer_len-equilibrate_len:], rate, rate[:buffer_len]])
        elif equilibrate_len > 0:
            rate = np.concatenate([rate[-equilibrate_len:], rate])
    else:
        t = np.arange(-buffer-equilibrate, duration + buffer + dt / 2., dt)
        rate = (max_rate - min_rate) * np.exp(-((t - peak_loc) / sigma) ** 2.) + min_rate
    return rate


def get_gaussian_prob_peak_locs(tuning_duration, pop_size, loc, sigma, depth, resolution=2, wrap_around=True):
    """

    :param tuning_duration: float
    :param pop_size: int
    :param loc: float
    :param sigma: float
    :param depth: float
    :param resolution: int
    :param wrap_around: bool
    :return: array of len pop_size * resolution
    """
    if wrap_around:
        peak_locs = np.linspace(0., tuning_duration, int(resolution * pop_size), endpoint=False)
        extended_peak_locs = np.concatenate([peak_locs - tuning_duration, peak_locs, peak_locs + tuning_duration])
        p_peak_locs = np.exp(-((extended_peak_locs - loc) / sigma) ** 2.)
        before = np.array(p_peak_locs[:len(peak_locs)])
        after = np.array(p_peak_locs[2 * len(peak_locs):])
        within = np.array(p_peak_locs[len(peak_locs):2 * len(peak_locs)])
        p_peak_locs = within[:len(peak_locs)] + before[:len(peak_locs)] + after[:len(peak_locs)]
    else:
        peak_locs = np.linspace(0., tuning_duration, int(resolution * pop_size), endpoint=False)
        p_peak_locs = np.exp(-((peak_locs - loc) / sigma) ** 2.)
    p_peak_locs /= np.sum(p_peak_locs)
    p_peak_locs *= depth
    p_peak_locs += 1. / len(peak_locs)
    p_peak_locs /= np.sum(p_peak_locs)

    return peak_locs, p_peak_locs


def get_pop_gid_ranges(pop_sizes):
    """

    :param pop_sizes: dict: {str: int}
    :return: dict: {str: tuple of int}
    """
    prev_gid = 0
    pop_gid_ranges = dict()
    for pop_name in pop_sizes:
        next_gid = prev_gid + pop_sizes[pop_name]
        pop_gid_ranges[pop_name] = (prev_gid, next_gid)
        prev_gid += pop_sizes[pop_name]
    return pop_gid_ranges


def infer_firing_rates_baks(spike_trains_dict, t, alpha, beta, pad_dur=0., wrap_around=False):
    """

    :param spike_trains_dict: nested dict: {pop_name: {gid: array} }
    :param t: array
    :param alpha: float
    :param beta: float
    :param pad_dur: float
    :param wrap_around: bool
    :return: dict of array
    """
    inferred_firing_rates = defaultdict(dict)
    for pop_name in spike_trains_dict:
        for gid, spike_train in viewitems(spike_trains_dict[pop_name]):
            if len(spike_train) > 0:
                this_inferred_rate = \
                    padded_baks(spike_train, t, alpha=alpha, beta=beta, pad_dur=pad_dur, wrap_around=wrap_around)
            else:
                this_inferred_rate = np.zeros_like(t)
            inferred_firing_rates[pop_name][gid] = this_inferred_rate

    return inferred_firing_rates


def get_binned_spike_count_dict(spike_times_dict, t):
    """

    :param spike_times_dict: nested dict: {pop_name: {gid: array} }
    :param t: array
    :return: nested dict
    """
    binned_spike_count_dict = dict()
    for pop_name in spike_times_dict:
        binned_spike_count_dict[pop_name] = dict()
        for gid in spike_times_dict[pop_name]:
            binned_spike_count_dict[pop_name][gid] = get_binned_spike_count(spike_times_dict[pop_name][gid], t)
    return binned_spike_count_dict


def get_firing_rate_from_binned_spike_count(binned_spike_count, bin_dur=20., smooth=None, wrap=False):
    """
    Use a gaussian filter to estimate firing rate from a binned spike count array.
    :param binned_spike_count: array of int
    :param bin_dur: float (ms)
    :param smooth: float (ms): standard deviation of temporal gaussian filter to smooth firing rate estimate
    :param wrap: bool
    :return: array
    """
    if smooth is None:
        return binned_spike_count / (bin_dur / 1000.)
    sigma = smooth / bin_dur
    if wrap:
        return gaussian_filter1d(binned_spike_count / (bin_dur / 1000.), sigma, mode='wrap')
    else:
        return gaussian_filter1d(binned_spike_count / (bin_dur / 1000.), sigma)


def get_firing_rates_from_binned_spike_count_dict(binned_spike_count_dict, bin_dur=20., smooth=None, wrap=False):
    """
    Use a gaussian filter to estimate firing rate for each cell population in a dict of binned spike count arrays.
    :param binned_spike_count_dict: dict: {pop_name: {gid: array with shape == (bins,) } }
    :param bin_dur: float (ms)
    :param smooth: float (ms): standard deviation of temporal gaussian filter to smooth firing rate estimate
    :param wrap: bool
    :return: tuple of dict
    """
    firing_rates_dict = {}
    for pop_name in binned_spike_count_dict:
        firing_rates_dict[pop_name] = {}
        for gid in binned_spike_count_dict[pop_name]:
            this_binned_spike_count = binned_spike_count_dict[pop_name][gid]
            firing_rates_dict[pop_name][gid] = \
                get_firing_rate_from_binned_spike_count(this_binned_spike_count, bin_dur, smooth, wrap)

    return firing_rates_dict


def get_firing_rates_from_binned_spike_count_matrix_dict(binned_spike_count_matrix_dict, bin_dur=20., smooth=None, 
                                                         wrap=False):
    """
    Use a gaussian filter to estimate firing rate for each cell population in a dict of binned spike count arrays.
    :param binned_spike_count_matrix_dict: dict: {pop_name: array with shape == (cells, bins) }
    :param bin_dur: float (ms)
    :param smooth: float (ms): standard deviation of temporal gaussian filter to smooth firing rate estimate
    :param wrap: bool
    :return: tuple of dict
    """
    firing_rates_matrix_dict = {}
    for pop_name in binned_spike_count_matrix_dict:
        firing_rates_matrix_dict[pop_name] = np.empty_like(binned_spike_count_matrix_dict[pop_name])
        for i in range(binned_spike_count_matrix_dict[pop_name].shape[0]):
            this_binned_spike_count = binned_spike_count_matrix_dict[pop_name][i,:]
            firing_rates_matrix_dict[pop_name][i,:] = \
                get_firing_rate_from_binned_spike_count(this_binned_spike_count, bin_dur, smooth, wrap)

    return firing_rates_matrix_dict


def find_nearest(arr, tt):
    arr = arr[arr >= tt[0]]
    arr = arr[arr <= tt[-1]]
    return np.searchsorted(tt, arr)


def padded_baks(spike_times, t, alpha, beta, pad_dur=500., wrap_around=False, plot=False):
    """
    Expects spike times in ms. Uses mirroring to pad the edges to avoid edge artifacts. Converts ms to sec for baks
    filtering, then returns the properly truncated estimated firing rate.
    :param spike_times: array
    :param t: array
    :param alpha: float
    :param beta: float
    :param pad_dur: float (ms)
    :param wrap_around: bool
    :param plot: bool
    :return: array
    """
    dt = t[1] - t[0]
    pad_dur = min(pad_dur, len(t)*dt)
    pad_len = int(pad_dur/dt)
    valid_spike_times = np.array(spike_times)
    valid_indexes = np.where((valid_spike_times >= t[0]) & (valid_spike_times <= t[-1]))[0]
    valid_spike_times = valid_spike_times[valid_indexes]
    if len(valid_spike_times) < 1:
        return np.zeros_like(t)
    if pad_len > 0:
        padded_spike_times = valid_spike_times
        r_pad_indexes = np.where((valid_spike_times > t[0]) & (valid_spike_times <= t[pad_len]))[0]
        l_pad_indexes = np.where((valid_spike_times >= t[-pad_len]) & (valid_spike_times < t[-1]))[0]
        if wrap_around:
            if len(r_pad_indexes) > 0:
                r_pad_spike_times = np.add(t[-1], np.subtract(valid_spike_times[r_pad_indexes], t[0]))
                padded_spike_times = np.append(padded_spike_times, r_pad_spike_times)
            if len(l_pad_indexes) > 0:
                l_pad_spike_times = np.add(t[0], np.subtract(valid_spike_times[l_pad_indexes], t[-1]+dt))
                padded_spike_times = np.append(l_pad_spike_times, padded_spike_times)
        else:
            if len(r_pad_indexes) > 0:
                r_pad_spike_times = np.add(t[0], np.subtract(t[0], valid_spike_times[r_pad_indexes])[::-1])
                padded_spike_times = np.append(r_pad_spike_times, padded_spike_times)
            if len(l_pad_indexes) > 0:
                l_pad_spike_times = np.add(t[-1], np.subtract(t[-1], valid_spike_times[l_pad_indexes])[::-1])
                padded_spike_times = np.append(padded_spike_times, l_pad_spike_times)
        padded_t = \
            np.concatenate((np.arange(t[0] - pad_dur, t[0], dt), t,
                            np.arange(t[-1] + dt, t[-1] + pad_dur + dt / 2., dt)))
        padded_rate, h = baks(padded_spike_times/1000., padded_t/1000., alpha, beta)
        if plot:
            fig = plt.figure()
            plt.plot(padded_t, padded_rate)
            plt.scatter(padded_spike_times, [0] * len(padded_spike_times), marker='.', color='k')
            fig.show()
        rate = padded_rate[pad_len:-pad_len]
    else:
        rate, h = baks(valid_spike_times/1000., t/1000., alpha, beta)
        if plot:
            fig = plt.figure()
            plt.plot(t, rate)
            plt.scatter(valid_spike_times, [0] * len(valid_spike_times), marker='.', color='k')
            fig.show()
    return rate


def get_binned_spike_count_orig(spike_times, t):
    """
    Convert spike times to a binned binary spike train
    :param spike_times: array (ms)
    :param t: array (ms)
    :return: array
    """
    binned_spikes = np.zeros_like(t)
    if len(spike_times) > 0:
        try:
            spike_indexes = []
            for spike_time in spike_times:
                if t[0] <= spike_time <= t[-1]:
                    spike_indexes.append(np.where(t <= spike_time)[0][-1])
        except Exception as e:
            print(spike_times)
            print(t)
            sys.stdout.flush()
            time.sleep(0.1)
            raise(e)
        binned_spikes[spike_indexes] = 1.
    return binned_spikes


def get_binned_spike_count(spike_times, edges):
    """
    Convert ordered spike times to a spike count array in bins specified by the provided edges array.
    :param spike_times: array (ms)
    :param edges: array (ms)
    :return: array
    """
    binned_spikes = np.zeros(len(edges) - 1, dtype='int32')
    valid_indexes = np.where((spike_times >= edges[0]) & (spike_times <= edges[-1]))[0]
    if len(valid_indexes) < 1:
        return binned_spikes
    spike_times = spike_times[valid_indexes]
    j = 0
    for i, bin_start in enumerate(edges[:-1]):
        if len(spike_times[j:]) < 1:
            break
        bin_end = edges[i+1]
        if i == len(edges) - 2:
            count = len(np.where((spike_times[j:] >= bin_start) & (spike_times[j:] <= bin_end))[0])
        else:
            count = len(np.where((spike_times[j:] >= bin_start) & (spike_times[j:] < bin_end))[0])
        j += count
        binned_spikes[i] = count

    return binned_spikes


def get_pop_mean_rate_from_binned_spike_count(binned_spike_count_dict, dt):
    """
    Calculate mean firing rate for each cell population.
    :param binned_spike_count_dict: nested dict of array
    :param dt: float (ms)
    :return: dict of array
    """
    pop_mean_rate_dict = dict()

    for pop_name in binned_spike_count_dict:
        pop_mean_rate_dict[pop_name] = \
            np.divide(np.mean(list(binned_spike_count_dict[pop_name].values()), axis=0), dt / 1000.)

    return pop_mean_rate_dict


def get_pop_activity_stats(firing_rates_dict, input_t, valid_t=None, threshold=2.):
    """
    Calculate firing rate statistics for each cell population.
    :param firing_rates_dict: nested dict of array
    :param input_t: array
    :param valid_t: array
    :param threshold: firing rate threshold for "active" cells: float (Hz)
    :return: tuple of dict
    """
    min_rate_dict = defaultdict(dict)
    peak_rate_dict = defaultdict(dict)
    mean_rate_active_cells_dict = dict()
    pop_fraction_active_dict = dict()
    mean_min_rate_dict = dict()
    mean_peak_rate_dict = dict()

    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]

    for pop_name in firing_rates_dict:
        this_active_cell_count = np.zeros_like(valid_t)
        this_summed_rate_active_cells = np.zeros_like(valid_t)
        for gid in firing_rates_dict[pop_name]:
            this_firing_rate = firing_rates_dict[pop_name][gid][valid_indexes]
            active_indexes = np.where(this_firing_rate >= threshold)[0]
            if len(active_indexes) > 0:
                this_active_cell_count[active_indexes] += 1.
                this_summed_rate_active_cells[active_indexes] += this_firing_rate[active_indexes]
                min_rate_dict[pop_name][gid] = np.min(this_firing_rate)
                peak_rate_dict[pop_name][gid] = np.max(this_firing_rate)

        active_indexes = np.where(this_active_cell_count > 0.)[0]
        if len(active_indexes) > 0:
            mean_rate_active_cells_dict[pop_name] = np.array(this_summed_rate_active_cells)
            mean_rate_active_cells_dict[pop_name][active_indexes] = \
                np.divide(this_summed_rate_active_cells[active_indexes], this_active_cell_count[active_indexes])
        else:
            mean_rate_active_cells_dict[pop_name] = np.zeros_like(valid_t)
        pop_fraction_active_dict[pop_name] = np.divide(this_active_cell_count, len(firing_rates_dict[pop_name]))
        if len(min_rate_dict[pop_name]) > 0:
            mean_min_rate_dict[pop_name] = np.mean(list(min_rate_dict[pop_name].values()))
            mean_peak_rate_dict[pop_name] = np.mean(list(peak_rate_dict[pop_name].values()))
        else:
            mean_min_rate_dict[pop_name] = 0.
            mean_peak_rate_dict[pop_name] = 0.

    return mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict


def analyze_selectivity_across_instances(centered_firing_rate_mean_dict_list):
    """
    Given a list of dicts containing data from different network instances, return dicts containing the mean and sem
    for the peak-centered firing rates of each population.
    :param centered_firing_rate_mean_dict_list: list of dict of array of float
    :return: tuple of dict
    """
    mean_centered_firing_rate_mean_dict = {}
    mean_centered_firing_rate_sem_dict = {}

    num_instances = len(centered_firing_rate_mean_dict_list)
    for centered_firing_rate_mean_dict_instance in centered_firing_rate_mean_dict_list:
        for pop_name in centered_firing_rate_mean_dict_instance:
            if pop_name not in mean_centered_firing_rate_mean_dict:
                mean_centered_firing_rate_mean_dict[pop_name] = []
            mean_centered_firing_rate_mean_dict[pop_name].append(centered_firing_rate_mean_dict_instance[pop_name])
    for pop_name in mean_centered_firing_rate_mean_dict:
        mean_centered_firing_rate_sem_dict[pop_name] = np.std(mean_centered_firing_rate_mean_dict[pop_name],
                                                              axis=0) / np.sqrt(num_instances)
        mean_centered_firing_rate_mean_dict[pop_name] = np.mean(mean_centered_firing_rate_mean_dict[pop_name], axis=0)

    return mean_centered_firing_rate_mean_dict, mean_centered_firing_rate_sem_dict


def get_trial_averaged_pop_activity_stats(mean_rate_active_cells_dict_list, pop_fraction_active_dict_list):
    """
    Given lists of dicts containing data across trials or network instances, return dicts containing the mean and sem
    for the fraction active and mean rate of active cells for each cell population.
    :param mean_rate_active_cells_dict_list: list of dict of array of float
    :param pop_fraction_active_dict_list: list of dict of array of float
    :return: tuple of dict
    """
    mean_rate_active_cells_mean_dict = {}
    mean_rate_active_cells_sem_dict = {}
    pop_fraction_active_mean_dict = {}
    pop_fraction_active_sem_dict = {}

    num_instances = len(mean_rate_active_cells_dict_list)
    for mean_rate_active_cells_dict_instance in mean_rate_active_cells_dict_list:
        for pop_name in mean_rate_active_cells_dict_instance:
            if pop_name not in mean_rate_active_cells_mean_dict:
                mean_rate_active_cells_mean_dict[pop_name] = []
            mean_rate_active_cells_mean_dict[pop_name].append(mean_rate_active_cells_dict_instance[pop_name])
    for pop_name in mean_rate_active_cells_mean_dict:
        mean_rate_active_cells_sem_dict[pop_name] = np.std(mean_rate_active_cells_mean_dict[pop_name],
                                                           axis=0) / np.sqrt(num_instances)
        mean_rate_active_cells_mean_dict[pop_name] = np.mean(mean_rate_active_cells_mean_dict[pop_name], axis=0)

    for pop_fraction_active_dict_instance in pop_fraction_active_dict_list:
        for pop_name in pop_fraction_active_dict_instance:
            if pop_name not in pop_fraction_active_mean_dict:
                pop_fraction_active_mean_dict[pop_name] = []
            pop_fraction_active_mean_dict[pop_name].append(pop_fraction_active_dict_instance[pop_name])
    for pop_name in pop_fraction_active_mean_dict:
        pop_fraction_active_sem_dict[pop_name] = np.std(pop_fraction_active_mean_dict[pop_name], axis=0) / np.sqrt(
            num_instances)
        pop_fraction_active_mean_dict[pop_name] = np.mean(pop_fraction_active_mean_dict[pop_name], axis=0)

    return mean_rate_active_cells_mean_dict, mean_rate_active_cells_sem_dict, pop_fraction_active_mean_dict, \
           pop_fraction_active_sem_dict


def plot_pop_activity_stats(binned_t_edges, mean_rate_active_cells_mean_dict, pop_fraction_active_mean_dict,
                            mean_rate_active_cells_sem_dict=None, pop_fraction_active_sem_dict=None, pop_order=None,
                            label_dict=None, color_dict=None):
    """
    Plot fraction active and mean rate of active cells for each cell population.
    :param binned_t_edges: array of float (ms)
    :param mean_rate_active_cells_mean_dict: dict of array of float
    :param pop_fraction_active_mean_dict: dict of array of float
    :param mean_rate_active_cells_sem_dict: dict of array of float
    :param pop_fraction_active_sem_dict: dict of array of float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    """
    if pop_order is None:
        pop_order = sorted(list(mean_rate_active_cells_mean_dict.keys()))

    binned_dt = binned_t_edges[1] - binned_t_edges[0]
    first_mean_rate = next(iter(mean_rate_active_cells_mean_dict.values()))
    if len(binned_t_edges) > len(first_mean_rate):
        binned_t = binned_t_edges[:-1] + binned_dt / 2.
    else:
        binned_t = np.copy(binned_t_edges)
    binned_t /= 1000.

    fig, axes = plt.subplots(1, 2, figsize=(7., 3.5))
    for pop_name in pop_order:
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        if color_dict is not None:
            color = color_dict[pop_name]
            axes[0].plot(binned_t, pop_fraction_active_mean_dict[pop_name], label=label, color=color, linewidth=2.)
            if pop_fraction_active_sem_dict is not None:
                axes[0].fill_between(binned_t,
                                     pop_fraction_active_mean_dict[pop_name] - pop_fraction_active_sem_dict[pop_name],
                                     pop_fraction_active_mean_dict[pop_name] + pop_fraction_active_sem_dict[pop_name],
                                     color=color,
                                     alpha=0.25, linewidth=0)
            axes[1].plot(binned_t, mean_rate_active_cells_mean_dict[pop_name], color=color, linewidth=2.)
            if mean_rate_active_cells_sem_dict is not None:
                axes[1].fill_between(binned_t,
                                     mean_rate_active_cells_mean_dict[pop_name] - mean_rate_active_cells_sem_dict[
                                         pop_name],
                                     mean_rate_active_cells_mean_dict[pop_name] + mean_rate_active_cells_sem_dict[
                                         pop_name], color=color,
                                     alpha=0.25, linewidth=0)
        else:
            axes[0].plot(binned_t, pop_fraction_active_mean_dict[pop_name], label=label, linewidth=2.)
            if pop_fraction_active_sem_dict is not None:
                axes[0].fill_between(binned_t,
                                     pop_fraction_active_mean_dict[pop_name] - pop_fraction_active_sem_dict[pop_name],
                                     pop_fraction_active_mean_dict[pop_name] + pop_fraction_active_sem_dict[pop_name],
                                     alpha=0.25, linewidth=0)
            axes[1].plot(binned_t, mean_rate_active_cells_mean_dict[pop_name], linewidth=2.)
            if mean_rate_active_cells_sem_dict is not None:
                axes[1].fill_between(binned_t,
                                     mean_rate_active_cells_mean_dict[pop_name] - mean_rate_active_cells_sem_dict[
                                         pop_name],
                                     mean_rate_active_cells_mean_dict[pop_name] + mean_rate_active_cells_sem_dict[
                                         pop_name],
                                     alpha=0.25, linewidth=0)

    axes[0].set_title('Active fraction\nof population', fontsize=mpl.rcParams['font.size'])
    axes[1].set_title('Mean firing rate\nof active cells', fontsize=mpl.rcParams['font.size'])
    axes[0].set_ylim((0., axes[0].get_ylim()[1]))
    axes[1].set_ylim((0., axes[1].get_ylim()[1]))
    axes[0].set_xlim((binned_t_edges[0] / 1000., binned_t_edges[-1] / 1000.))
    axes[1].set_xlim((binned_t_edges[0] / 1000., binned_t_edges[-1] / 1000.))
    axes[0].set_ylabel('Fraction')
    axes[1].set_ylabel('Firing rate (Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[1].set_xlabel('Time (s)')
    axes[0].set_xticks(np.arange(0., axes[0].get_xlim()[1] + binned_dt / 1000. / 2., 1.))
    axes[1].set_xticks(np.arange(0., axes[1].get_xlim()[1] + binned_dt / 1000. / 2., 1.))
    axes[0].legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'], handlelength=1)
    clean_axes(axes)
    fig.tight_layout()
    fig.show()


def get_butter_bandpass_filter(filter_band, sampling_rate, order, filter_label='', plot=False):
    """

    :param filter_band: list of float
    :param sampling_rate: float
    :param order: int
    :param filter_label: str
    :param plot: bool
    :return: array
    """
    nyq = 0.5 * sampling_rate
    normalized_band = np.divide(filter_band, nyq)
    sos = butter(order, normalized_band, analog=False, btype='band', output='sos')
    if plot:
        fig = plt.figure()
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((sampling_rate * 0.5 / np.pi) * w, abs(h), c='k')
        plt.plot([0, 0.5 * sampling_rate], [np.sqrt(0.5), np.sqrt(0.5)], '--', c='grey')
        plt.title('%s bandpass filter (%.1f:%.1f Hz), Order: %i' %
                  (filter_label, min(filter_band), max(filter_band), order), fontsize=mpl.rcParams['font.size'])
        plt.xlabel('Frequency (Hz)')
        bandwidth = max(filter_band) - min(filter_band)
        plt.xlim(max(0., min(filter_band) - bandwidth / 2.), min(nyq, max(filter_band) + bandwidth / 2.))
        plt.ylabel('Gain')
        plt.grid(True)
        fig.show()

    return sos


def get_butter_lowpass_filter(cutoff_freq, sampling_rate, order, filter_label='', plot=False):
    """

    :param cutoff_freq: float
    :param sampling_rate: float
    :param order: int
    :param filter_label: str
    :param plot: bool
    :return: array
    """
    nyq = 0.5 * sampling_rate
    normalized_cutoff_freq = cutoff_freq / nyq
    sos = butter(order, normalized_cutoff_freq, analog=False, btype='low', output='sos')
    if plot:
        fig = plt.figure()
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((sampling_rate * 0.5 / np.pi) * w, abs(h), c='k')
        plt.plot([0, 0.5 * sampling_rate], [np.sqrt(0.5), np.sqrt(0.5)], '--', c='grey')
        plt.title('%s low-pass filter (%.1f Hz), Order: %i' %
                  (filter_label, cutoff_freq, order), fontsize=mpl.rcParams['font.size'])
        plt.xlabel('Frequency (Hz)')
        plt.xlim(0., cutoff_freq * 1.5)
        plt.ylabel('Gain')
        plt.grid(True)
        fig.show()

    return sos


def PSTI(f, power, band=None, verbose=False):
    """
    'Power spectral tuning index'. Signal and noise partitioned as a quantile of the power distribution. Standard
    deviation in the frequency domain is normalized to the bandwidth. Resulting frequency tuning index is proportional
    to the amplitude ratio of signal power to noise power, and inversely proportional to the standard deviation in the
    frequency domain.
    :param f: array of float; frequency (Hz)
    :param power: array of float; power spectral density (units^2/Hz)
    :param band: tuple of float
    :param verbose: bool
    :return: float
    """
    if band is None:
        band = (np.min(f), np.max(f))
    band_indexes = np.where((f >= band[0]) & (f <= band[1]))[0]
    if len(band_indexes) == 0:
        raise ValueError('PSTI: sample does not contain specified band')
    power_std = np.std(power[band_indexes])
    if power_std == 0.:
        return 0.
    bandwidth = band[1] - band[0]
    min_power = np.min(power[band_indexes])
    if min_power < 0.:
        raise ValueError('PTSI: power density array must be non-negative')
    if np.max(power[band_indexes]) - min_power == 0.:
        return 0.

    half_width_indexes = get_mass_index(power[band_indexes], 0.25), get_mass_index(power[band_indexes], 0.75)
    if half_width_indexes[0] == half_width_indexes[1]:
        norm_f_signal_width = (f[band_indexes][1] - f[band_indexes][0]) / bandwidth
    else:
        norm_f_signal_width = (f[band_indexes][half_width_indexes[1]] - f[band_indexes][half_width_indexes[0]]) / \
                              bandwidth

    top_quartile_indexes = get_mass_index(power[band_indexes], 0.375), get_mass_index(power[band_indexes], 0.625)
    if top_quartile_indexes[0] == top_quartile_indexes[1]:
        signal_mean = power[band_indexes][top_quartile_indexes[0]]
    else:
        signal_indexes = np.arange(top_quartile_indexes[0], top_quartile_indexes[1], 1)
        signal_mean = np.mean(power[band_indexes][signal_indexes])
    if signal_mean == 0.:
        return 0.

    bottom_quartile_indexes = get_mass_index(power[band_indexes], 0.125), get_mass_index(power[band_indexes], 0.875)
    noise_indexes = np.concatenate([np.arange(0, bottom_quartile_indexes[0], 1),
                                    np.arange(bottom_quartile_indexes[1], len(band_indexes), 1)])
    noise_mean = np.mean(power[band_indexes][noise_indexes])

    if verbose:
        print('PSTI: delta_power: %.5f; power_std: %.5f, norm_f_signal_width: %.5f, half_width_edges: [%.5f, %.5f]' %
              (signal_mean - noise_mean, power_std, norm_f_signal_width, f[band_indexes][half_width_indexes[0]],
               f[band_indexes][half_width_indexes[1]]))
        sys.stdout.flush()

    this_PSTI = (signal_mean - noise_mean) / power_std / norm_f_signal_width / 2.
    return this_PSTI


def get_freq_tuning_stats(fft_f, fft_power, filter_band, buffered_filter_band=None, bins=100, verbose=False):
    """

    :param fft_f: array (Hz)
    :param fft_power: array (units**2/Hz)
    :param filter_band: list of float (Hz)
    :param buffered_filter_band: list of float (Hz)
    :param bins: int
    :param verbose: bool
    :return: tuple of array
    """
    if buffered_filter_band is not None:
        f = np.linspace(buffered_filter_band[0], buffered_filter_band[1], bins)
    else:
        f = np.linspace(filter_band[0], filter_band[1], bins)
    power = np.interp(f, fft_f, fft_power)

    com_index = get_mass_index(power, 0.5)
    if com_index is None:
        centroid_freq = 0.
    else:
        centroid_freq = f[com_index]
    if centroid_freq == 0. or centroid_freq < filter_band[0] or centroid_freq > filter_band[1]:
        freq_tuning_index = 0.
    else:
        if buffered_filter_band is not None:
            freq_tuning_index = PSTI(f, power, band=buffered_filter_band, verbose=verbose)
        else:
            freq_tuning_index = PSTI(f, power, band=filter_band, verbose=verbose)

    return f, power, centroid_freq, freq_tuning_index


def get_bandpass_filtered_signal_stats(signal, input_t, sos, filter_band, output_t=None, pad=True, filter_label='',
                                       signal_label='', verbose=False):
    """

    :param signal: array
    :param input_t: array (ms)
    :param sos: array
    :param filter_band: list of float (Hz)
    :param output_t: array (ms)
    :param pad: bool
    :param filter_label: str
    :param signal_label: str
    :param verbose: bool
    :return: tuple of array
    """
    if np.all(signal == 0.):
        if verbose > 0:
            print('%s\n%s bandpass filter (%.1f:%.1f Hz); Failed - no signal' %
                  (signal_label, filter_label, min(filter_band), max(filter_band)))
            sys.stdout.flush()
        return np.zeros_like(output_t), np.zeros_like(output_t), np.zeros_like(output_t), 0.
    dt = input_t[1] - input_t[0]  # ms

    if pad:
        pad_dur = min(10. * 1000. / np.min(filter_band), len(input_t) * dt)  # ms
        pad_len = min(int(pad_dur / dt), len(input_t) - 1)
        intermediate_signal = get_mirror_padded_signal(signal, pad_len)
    else:
        intermediate_signal = np.array(signal)

    filtered_signal = sosfiltfilt(sos, intermediate_signal)
    envelope = np.abs(hilbert(filtered_signal))
    if pad:
        filtered_signal = filtered_signal[pad_len:-pad_len]
        envelope = envelope[pad_len:-pad_len]

    if output_t is not None:
        output_signal = np.interp(output_t, input_t, signal)
        filtered_signal = np.interp(output_t, input_t, filtered_signal)
        envelope = np.interp(output_t, input_t, envelope)
    else:
        output_signal = signal

    mean_envelope = np.mean(envelope)
    mean_signal = np.mean(output_signal)
    if mean_signal == 0.:
        envelope_ratio = 0.
    else:
        envelope_ratio = mean_envelope / mean_signal

    return output_signal, filtered_signal, envelope, envelope_ratio


def get_pop_bandpass_filtered_signal_stats(signal_dict, filter_band_dict, input_t, valid_t=None, output_t=None,
                                           order=15, pad=True, bins=100, filter_order=None, filter_label_dict=None,
                                           filter_color_dict=None, filter_xlim_dict=None, pop_order=None,
                                           label_dict=None, color_dict=None, plot=False, verbose=False):
    """

    :param signal_dict: array
    :param filter_band_dict: dict: {filter_label (str): list of float (Hz) }
    :param input_t: array (ms)
    :param valid_t: array (ms)
    :param output_t: array (ms)
    :param order: int
    :param pad: bool
    :param bins: int
    :param filter_order: list of str
    :param filter_label_dict: dict
    :param filter_color_dict: dict
    :param filter_xlim_dict: dict of tuple of float
    :param pop_order: list of str
    :param label_dict: dict
    :param color_dict: dict
    :param plot: bool
    :param verbose: bool
    :return: tuple of dict
    """
    dt = input_t[1] - input_t[0]  # ms
    sampling_rate = 1000. / dt  # Hz
    output_signal = {}
    filtered_signal = {}
    fft_f_dict = {}
    fft_power_dict = {}
    psd_f_dict = {}
    psd_power_dict = {}
    envelope_dict = {}
    envelope_ratio_dict = {}
    centroid_freq_dict = {}
    freq_tuning_index_dict = {}
    sos_dict = {}
    buffered_filter_band_dict = {}
    psd_power_range = {}

    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]

    for filter_label, filter_band in viewitems(filter_band_dict):
        output_signal[filter_label] = {}
        filtered_signal[filter_label] = {}
        psd_f_dict[filter_label] = {}
        psd_power_dict[filter_label] = {}
        envelope_dict[filter_label] = {}
        envelope_ratio_dict[filter_label] = {}
        centroid_freq_dict[filter_label] = {}
        freq_tuning_index_dict[filter_label] = {}
        sos_dict[filter_label] = get_butter_bandpass_filter(filter_band, sampling_rate, filter_label=filter_label,
                                                            order=order, plot=plot)
        buffered_filter_band_dict[filter_label] = [filter_band[0] / 2., 2. * filter_band[1]]

    for pop_name in signal_dict:
        signal = signal_dict[pop_name][valid_indexes]
        fft_f_dict[pop_name], fft_power_dict[pop_name] = periodogram(signal, fs=sampling_rate)

        for filter_label, filter_band in viewitems(filter_band_dict):
            output_signal[filter_label][pop_name], filtered_signal[filter_label][pop_name], \
            envelope_dict[filter_label][pop_name], envelope_ratio_dict[filter_label][pop_name] = \
                get_bandpass_filtered_signal_stats(signal, valid_t, sos_dict[filter_label], filter_band,
                                                   output_t=output_t, pad=pad, filter_label=filter_label,
                                                   signal_label='Population: %s' % pop_name, verbose=verbose)
            psd_f_dict[filter_label][pop_name], psd_power_dict[filter_label][pop_name], \
            centroid_freq_dict[filter_label][pop_name], freq_tuning_index_dict[filter_label][pop_name] = \
                get_freq_tuning_stats(fft_f_dict[pop_name], fft_power_dict[pop_name], filter_band,
                                      buffered_filter_band=buffered_filter_band_dict[filter_label], bins=bins,
                                      verbose=verbose)
            if filter_label not in psd_power_range:
                psd_power_range[filter_label] = [np.min(psd_power_dict[filter_label][pop_name]),
                                                 np.max(psd_power_dict[filter_label][pop_name])]
            else:
                psd_power_range[filter_label][0] = min(np.min(psd_power_dict[filter_label][pop_name]),
                                                       psd_power_range[filter_label][0])
                psd_power_range[filter_label][1] = max(np.max(psd_power_dict[filter_label][pop_name]),
                                                       psd_power_range[filter_label][1])

    if plot:
        for filter_label, filter_band in viewitems(filter_band_dict):
            for pop_name in output_signal[filter_label]:
                plot_bandpass_filtered_signal_summary(output_t, output_signal[filter_label][pop_name],
                                                      filtered_signal[filter_label][pop_name], filter_band,
                                                      envelope_dict[filter_label][pop_name],
                                                      psd_f_dict[filter_label][pop_name],
                                                      psd_power_dict[filter_label][pop_name],
                                                      centroid_freq_dict[filter_label][pop_name],
                                                      freq_tuning_index_dict[filter_label][pop_name],
                                                      psd_power_range=psd_power_range[filter_label],
                                                      buffered_filter_band=buffered_filter_band_dict[filter_label],
                                                      signal_label='Population: %s' % pop_name,
                                                      filter_label=filter_label, axis_label='Firing rate', units='Hz')
        plot_rhythmicity_traces(output_t, output_signal, filtered_signal, filter_band_dict, filter_order=filter_order,
                                 filter_label_dict=filter_label_dict, filter_color_dict=filter_color_dict,
                                 filter_xlim_dict=filter_xlim_dict, pop_order=pop_order, label_dict=label_dict,
                                 color_dict=color_dict)

    return fft_f_dict, fft_power_dict, psd_f_dict, psd_power_dict, envelope_dict, envelope_ratio_dict, \
           centroid_freq_dict, freq_tuning_index_dict


def plot_compare_binned_spike_counts(binned_spike_count_dict, t, xlim=None, pop_order=None, label_dict=None,
                                     color_dict=None):
    """
    Superimposed population spike counts from multiple specified populations.
    :param binned_spike_count_dict: array
    :param t: array (ms)
    :param xlim: tuple of float
    :param pop_order: list of str
    :param label_dict: dict
    :param color_dict: dict
    """
    if pop_order is None:
        pop_order = sorted(list(signal_dict.keys()))
    if label_dict is None:
        label_dict = {}
        for pop_name in pop_order:
            label_dict[pop_name] = pop_name

    fig, axis = plt.subplots(figsize=(4.5, 3.5))
    for pop_name in pop_order:
        if color_dict is None or pop_name not in color_dict:
            color = None
        else:
            color = color_dict[pop_name]
        pop_spike_count = np.sum(list(binned_spike_count_dict[pop_name].values()), axis=0)
        axis.plot(t, pop_spike_count, color=color, alpha=0.5, label=label_dict[pop_name])
    if xlim is not None:
        axis.set_xlim(xlim)
    axis.set_xlabel('Time (ms)')
    axis.set_ylabel('Population\nspike count')
    axis.legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'], handlelength=1)
    clean_axes(axis)
    fig.tight_layout()
    fig.show()


def plot_bandpass_filtered_signal_summary(t, signal, filtered_signal, filter_band, envelope, psd_f, psd_power,
                                          centroid_freq, freq_tuning_index, psd_power_range=None,
                                          buffered_filter_band=None, signal_label='', filter_label='',
                                          axis_label='Amplitude', units='a.u.'):
    """
    :param t: array
    :param signal: array
    :param filtered_signal: array
    :param filter_band: list of float (Hz)
    :param envelope: array
    :param psd_f: array
    :param psd_power: array
    :param centroid_freq: float
    :param freq_tuning_index: float
    :param psd_power_range: list
    :param buffered_filter_band: list of float (Hz)
    :param signal_label: str
    :param filter_label: str
    :param axis_label: str
    :param units: str
    :return: tuple of array
    """
    mean_envelope = np.mean(envelope)
    mean_signal = np.mean(signal)
    if mean_signal == 0.:
        envelope_ratio = 0.
    else:
        envelope_ratio = mean_envelope / mean_signal

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7))
    axes[0][0].plot(t, np.subtract(signal, np.mean(signal)), c='grey', alpha=0.5, label='Original signal')
    axes[0][0].plot(t, filtered_signal, c='r', label='Filtered signal', alpha=0.5)
    axes[0][1].plot(t, signal, label='Original signal', c='grey', alpha=0.5, zorder=2)
    axes[0][1].plot(t, np.ones_like(t) * mean_signal, c='k', zorder=1)
    axes[0][1].plot(t, envelope, label='Envelope amplitude', c='r', alpha=0.5, zorder=2)
    axes[0][1].plot(t, np.ones_like(t) * mean_envelope, c='darkred', zorder=0)
    axes[0][0].set_ylabel('%s (%s)\n(mean subtracted)' % (axis_label, units))
    axes[0][1].set_ylabel('%s (%s)' % (axis_label, units))
    box = axes[0][0].get_position()
    axes[0][0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
    axes[0][0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, framealpha=0.5, handlelength=1)
    axes[0][0].set_xlabel('Time (ms)')
    box = axes[0][1].get_position()
    axes[0][1].set_position([box.x0, box.y0, box.width, box.height * 0.8])
    axes[0][1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, framealpha=0.5, handlelength=1)
    axes[0][1].set_xlabel('Time (ms)')

    axes[1][0].plot(psd_f, psd_power, c='k')
    # axes[1][0].semilogy(psd_f, psd_power, c='k')
    axes[1][0].set_xlabel('Frequency (Hz)')
    axes[1][0].set_ylabel('Spectral density\n(units$^{2}$/Hz)')
    if buffered_filter_band is not None:
        axes[1][0].set_xlim(min(buffered_filter_band), max(buffered_filter_band))
    else:
        axes[1][0].set_xlim(min(filter_band), max(filter_band))
    if psd_power_range is not None:
        axes[1][0].set_ylim(psd_power_range[0], psd_power_range[1])

    clean_axes(axes)
    fig.suptitle('%s: %s bandpass filter (%.1f:%.1f Hz)\nEnvelope ratio: %.3f; Centroid freq: %.3f Hz\n'
                 'Frequency tuning index: %.3f' % (signal_label, filter_label, min(filter_band), max(filter_band),
                                                   envelope_ratio, centroid_freq, freq_tuning_index),
                 fontsize=mpl.rcParams['font.size'])
    fig.tight_layout()
    fig.subplots_adjust(top=0.75, hspace=0.3)
    fig.show()


def plot_rhythmicity_traces(t, signal_dict, filtered_signal_dict, filter_band_dict, filter_order=None,
                            filter_label_dict=None, filter_color_dict=None, filter_xlim_dict=None, pop_order=None,
                            label_dict=None, color_dict=None):
    """

    :param t: array of float
    :param signal_dict: nested dict: {filter_label: {pop_name: array of float} }
    :param filtered_signal_dict: nested dict: {filter_label: {pop_name: array of float} }
    :param filter_band_dict: dict of tuple of float
    :param filter_order: list of str
    :param filter_label_dict: dict
    :param filter_color_dict: dict
    :param filter_xlim_dict: dict of tuple of float
    :param pop_order: list of str
    :param label_dict: dict
    :param color_dict: dict
    """
    num_filters = len(filter_band_dict)
    if pop_order is None:
        pop_order = sorted(list(next(iter(signal_dict.values())).keys()))
    num_pops = len(pop_order)
    num_rows = 1 + num_filters
    if filter_order is None:
        filter_order = sorted(list(filter_band_dict.keys()))

    fig, axes = plt.subplots(num_rows, num_pops, figsize=(4. * num_pops, 1.7 * num_rows))
    for i, pop_name in enumerate(pop_order):
        first_signal = next(iter(signal_dict.values()))[pop_name]
        axes[0][i].plot(t, np.subtract(first_signal, np.mean(first_signal)), c='grey', alpha=0.5)
        ylim = list(axes[0][i].get_ylim())
        if i == 0:
            axes[0, 0].set_ylabel('Population\naverage\nfiring rate')
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        axes[0][i].set_title(label, fontsize=mpl.rcParams['font.size'])
        axes[0][i].set_xlim((t[0], t[-1]))
        for j, filter_label in enumerate(filter_order):
            filter_band = filter_band_dict[filter_label]
            if filter_xlim_dict is not None:
                xlim = filter_xlim_dict[filter_label]
            else:
                min_band_freq = filter_band[0]
                bin_dur = 1. / min_band_freq * 1000.
                xlim = (5. * bin_dur, 10 * bin_dur)
            axes[1 + j][i].plot(t, np.subtract(signal_dict[filter_label][pop_name],
                                               np.mean(signal_dict[filter_label][pop_name])), \
                                c='grey', alpha=0.5)
            if filter_color_dict is not None:
                axes[1 + j][i].plot(t, filtered_signal_dict[filter_label][pop_name], c=filter_color_dict[filter_label],
                                    alpha=0.5)
            else:
                axes[1 + j][i].plot(t, filtered_signal_dict[filter_label][pop_name], alpha=0.5)
            axes[1 + j][i].set_xlim(xlim)
            ylim[0] = min(ylim[0], axes[1 + j][i].get_ylim()[0])
            ylim[1] = max(ylim[1], axes[1 + j][i].get_ylim()[1])
            if i == 0:
                if filter_label_dict is not None:
                    label = filter_label_dict[filter_label]
                else:
                    label = filter_label
                axes[1 + j][i].set_ylabel('%s\n(%i - %i Hz)' % (label, min(filter_band), max(filter_band)))
        for j in range(len(axes)):
            axes[j][i].set_ylim(ylim)

    clean_axes(axes)
    fig.tight_layout()
    fig.subplots_adjust(top=0.75, hspace=0.3)
    fig.show()


def get_lowpass_filtered_signal_stats(signal, input_t, sos, cutoff_freq, output_t=None, bins=100, signal_label='',
                                      filter_label='', axis_label='Amplitude', units='a.u.', pad=True, pad_len=None,
                                      plot=False, verbose=False):
    """

    :param signal: array
    :param input_t: array (ms)
    :param sos: array
    :param cutoff_freq:  float (Hz)
    :param output_t: array (ms)
    :param bins: number of frequency bins to compute in band
    :param signal_label: str
    :param filter_label: str
    :param axis_label: str
    :param units: str
    :param pad: bool
    :param pad_len: int
    :param plot: bool
    :param verbose: bool
    :return: tuple of array
    """
    if np.all(signal == 0.):
        if verbose > 0:
            print('%s\n%s low-pass filter (%.1f Hz); Failed - no signal' %
                  (signal_label, filter_label, cutoff_freq))
            sys.stdout.flush()
        return signal, np.zeros_like(signal), 0., 0., 0.
    dt = input_t[1] - input_t[0]  # ms
    fs = 1000. / dt

    nfft = int(fs * bins / cutoff_freq)

    if pad and pad_len is None:
        pad_dur = min(10. * 1000. / cutoff_freq, len(input_t) * dt)  # ms
        pad_len = min(int(pad_dur / dt), len(input_t) - 1)
    if pad:
        padded_signal = get_mirror_padded_signal(signal, pad_len)
    else:
        padded_signal = np.array(signal)

    filtered_padded_signal = sosfiltfilt(sos, padded_signal)
    filtered_signal = filtered_padded_signal[pad_len:-pad_len]
    padded_envelope = np.abs(hilbert(filtered_padded_signal))
    envelope = padded_envelope[pad_len:-pad_len]

    f, power = periodogram(filtered_signal, fs=fs, nfft=nfft)

    com_index = get_mass_index(power, 0.5)
    if com_index is None:
        centroid_freq = 0.
    else:
        centroid_freq = f[com_index]
    if centroid_freq == 0. or centroid_freq > cutoff_freq:
        freq_tuning_index = 0.
    else:
        freq_tuning_index = PSTI(f, power, band=[0., cutoff_freq], verbose=verbose)

    if output_t is None:
        t = input_t
    else:
        t = output_t
        signal = np.interp(output_t, input_t, signal)
        filtered_signal = np.interp(output_t, input_t, filtered_signal)
        envelope = np.interp(output_t, input_t, envelope)

    mean_envelope = np.mean(envelope)
    mean_signal = np.mean(signal)
    if mean_signal == 0.:
        envelope_ratio = 0.
    else:
        envelope_ratio = mean_envelope / mean_signal

    if plot:
        fig, axes = plt.subplots(2,2, figsize=(8.5,7))
        axes[0][0].plot(t, np.subtract(signal, np.mean(signal)), c='grey', alpha=0.5, label='Original signal')
        axes[0][0].plot(t, filtered_signal, c='r', label='Filtered signal', alpha=0.5)
        axes[0][1].plot(t, signal, label='Original signal', c='grey', alpha=0.5, zorder=2)
        axes[0][1].plot(t, np.ones_like(t) * mean_signal, c='k', zorder=1)
        axes[0][1].plot(t, envelope, label='Envelope amplitude', c='r', alpha=0.5, zorder=2)
        axes[0][1].plot(t, np.ones_like(t) * mean_envelope, c='darkred', zorder=0)
        axes[0][0].set_ylabel('%s\n(mean subtracted) (%s)' % (axis_label, units))
        axes[0][1].set_ylabel('%s (%s)' % (axis_label, units))
        box = axes[0][0].get_position()
        axes[0][0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        axes[0][0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, framealpha=0.5)
        axes[0][0].set_xlabel('Time (ms)')
        box = axes[0][1].get_position()
        axes[0][1].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        axes[0][1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, framealpha=0.5)
        axes[0][1].set_xlabel('Time (ms)')

        axes[1][0].plot(f, power, c='k')
        axes[1][0].set_xlabel('Frequency (Hz)')
        axes[1][0].set_ylabel('Spectral density\n(units$^{2}$/Hz)')
        axes[1][0].set_xlim(0., cutoff_freq)

        clean_axes(axes)
        fig.suptitle('%s: %s low-pass filter (%.1f Hz)\nEnvelope ratio: %.3f; Centroid freq: %.3f Hz\n'
                     'Frequency tuning index: %.3f' % (signal_label, filter_label, cutoff_freq,
                                                       envelope_ratio, centroid_freq, freq_tuning_index),
                     fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.75, hspace=0.3)
        fig.show()

    return filtered_signal, envelope, envelope_ratio, centroid_freq, freq_tuning_index


def get_pop_lowpass_filtered_signal_stats(signal_dict, filter_band_dict, input_t, output_t=None, order=15, plot=False,
                                           verbose=False):
    """

    :param signal_dict: array
    :param filter_band_dict: dict: {filter_label (str): list of float (Hz) }
    :param input_t: array (ms)
    :param output_t: array (ms)
    :param order: int
    :param plot: bool
    :param verbose: bool
    :return: tuple of dict
    """
    dt = input_t[1] - input_t[0]  # ms
    sampling_rate = 1000. / dt  # Hz
    filtered_signal_dict = {}
    envelope_dict = {}
    envelope_ratio_dict = {}
    centroid_freq_dict = {}
    freq_tuning_index_dict = {}
    for filter_label, cutoff_freq in viewitems(filter_band_dict):
        filtered_signal_dict[filter_label] = {}
        envelope_dict[filter_label] = {}
        envelope_ratio_dict[filter_label] = {}
        centroid_freq_dict[filter_label] = {}
        freq_tuning_index_dict[filter_label] = {}
        sos = get_butter_lowpass_filter(cutoff_freq, sampling_rate, filter_label=filter_label, order=order, plot=plot)

        for pop_name in signal_dict:
            signal = signal_dict[pop_name]
            filtered_signal_dict[filter_label][pop_name], envelope_dict[filter_label][pop_name], \
            envelope_ratio_dict[filter_label][pop_name], centroid_freq_dict[filter_label][pop_name], \
            freq_tuning_index_dict[filter_label][pop_name] = \
                get_lowpass_filtered_signal_stats(signal, input_t, sos, cutoff_freq, output_t=output_t,
                                                   signal_label='Population: %s' % pop_name, filter_label=filter_label,
                                                   axis_label='Firing rate', units='Hz', plot=plot, verbose=verbose)

    return filtered_signal_dict, envelope_dict, envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict


def plot_heatmap_from_matrix(data, xticks=None, xtick_labels=None, yticks=None, ytick_labels=None, ax=None,
                             cbar_kw={}, cbar_label="", rotate_xtick_labels=False, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        xticks : A list or array of length <=M with xtick locations
        xtick_labels : A list or array of length <=M with xtick labels
        yticks : A list or array of length <=N with ytick locations
        ytick_labels : A list or array of length <=N with ytick labels
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbar_label  : The label for the colorbar
        rotate_xtick_labels    : bool; whether to rotate xtick labels
    All other arguments are directly passed on to the imshow call.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    if xticks is not None:
        ax.set_xticks(xticks)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)

    if rotate_xtick_labels:
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")


def get_mass_index(signal, fraction=0.5, subtract_min=True):
    """
    Return the index of the center of mass of a signal, or None if the signal is mean zero. By default searches for
    area above the signal minimum.
    :param signal: array
    :param fraction: float in [0, 1]
    :param subtract_min: bool
    :return: int
    """
    if fraction < 0. or fraction > 1.:
        raise ValueError('get_mass_index: value of mass fraction must be between 0 and 1')
    if subtract_min:
        this_signal = np.subtract(signal, np.min(signal))
    else:
        this_signal = np.array(signal)
    cumsum = np.cumsum(this_signal)
    if cumsum[-1] == 0.:
        return None
    normalized_cumsum = cumsum / cumsum[-1]
    return np.argwhere(normalized_cumsum >= fraction)[0][0]


def get_mirror_padded_signal(signal, pad_len):
    """
    Pads the ends of the signal by mirroring without duplicating the end points.
    :param signal: array
    :param pad_len: int
    :return: array
    """
    mirror_beginning = signal[:pad_len][::-1]
    mirror_end = signal[-pad_len:][::-1]
    padded_signal = np.concatenate((mirror_beginning, signal, mirror_end))
    return padded_signal


def get_mirror_padded_time_series(t, pad_len):
    """
    Pads the ends of a time series by mirroring without duplicating the end points.
    :param t: array
    :param pad_len: int
    :return: array
    """
    dt = t[1] - t[0]
    t_end = len(t) * dt
    padded_t = np.concatenate((
        np.subtract(t[0] - dt, t[:pad_len])[::-1], t, np.add(t[-1], np.subtract(t_end, t[-pad_len:])[::-1])))
    return padded_t


def plot_inferred_spike_rates(spike_times_dict, firing_rates_dict, input_t, valid_t=None, active_rate_threshold=1.,
                              rows=3, cols=4, pop_names=None):
    """

    :param spike_times_dict: dict of array
    :param firing_rates_dict: dict of array
    :param input_t: array
    :param valid_t: array
    :param active_rate_threshold: float
    :param rows: int
    :param cols: int
    :param pop_names: list of str
    """
    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    if pop_names is None:
        pop_names = sorted(list(spike_times_dict.keys()))
    for pop_name in pop_names:
        active_gid_range = []
        for gid, rate in viewitems(firing_rates_dict[pop_name]):
            if np.max(rate) >= active_rate_threshold:
                active_gid_range.append(gid)
        if len(active_gid_range) > 0:
            fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(cols*3, rows*3))
            for j in range(cols):
                axes[rows-1][j].set_xlabel('Time (ms)')
            for i in range(rows):
                axes[i][0].set_ylabel('Firing rate (Hz)')
            gid_sample = random.sample(active_gid_range, min(len(active_gid_range), rows * cols))
            for i, gid in enumerate(gid_sample):
                this_spike_times = spike_times_dict[pop_name][gid]
                valid_spike_indexes = np.where((this_spike_times >= valid_t[0]) & (this_spike_times <= valid_t[-1]))[0]
                inferred_rate = firing_rates_dict[pop_name][gid][valid_indexes]
                row = i // cols
                col = i % cols
                axes[row][col].plot(valid_t, inferred_rate, label='Rate')
                axes[row][col].scatter(this_spike_times[valid_spike_indexes], [1.] * len(valid_spike_indexes),
                                       marker='.', color='k', label='Spikes')
                axes[row][col].set_title('gid: {}'.format(gid), fontsize=mpl.rcParams['font.size'])
            axes[0][cols-1].legend(loc='center left', frameon=False, framealpha=0.5, bbox_to_anchor=(1., 0.5))
            clean_axes(axes)
            fig.suptitle('Inferred spike rates: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
            fig.tight_layout()
            fig.subplots_adjust(top=0.9, right=0.9)
            fig.show()
        else:
            print('plot_inferred_spike_rates: no active cells for population: %s' % pop_name)
            sys.stdout.flush()


def plot_voltage_traces(voltage_rec_dict, input_t, valid_t=None, spike_times_dict=None, rows=3, cols=4, pop_names=None,
                        xlim=None):
    """

    :param voltage_rec_dict: dict of array
    :param input_t: array
    :param valid_t: array
    :param spike_times_dict: nested dict of array
    :param rows: int
    :param cols: int
    :param pop_names: list of str
    :param xlim: tuple of float
    """
    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    if xlim is None:
        xlim = (max(0., valid_t[0]), min(1500., valid_t[-1]))
    if pop_names is None:
        pop_names = sorted(list(voltage_rec_dict.keys()))
    for pop_name in pop_names:
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(cols*3+0.75, rows*3))
        for j in range(cols):
            axes[rows - 1][j].set_xlabel('Time (ms)')
        for i in range(rows):
            axes[i][0].set_ylabel('Voltage (mV)')
        this_gid_range = list(voltage_rec_dict[pop_name].keys())
        gid_sample = random.sample(this_gid_range, min(len(this_gid_range), rows * cols))
        for i, gid in enumerate(gid_sample):
            rec = np.interp(valid_t, input_t, voltage_rec_dict[pop_name][gid])
            row = i // cols
            col = i % cols
            axes[row][col].plot(valid_t, rec, label='Vm', c='k', linewidth=0.75)
            if spike_times_dict is not None and pop_name in spike_times_dict and gid in spike_times_dict[pop_name]:
                binned_spike_indexes = find_nearest(spike_times_dict[pop_name][gid], valid_t)
                axes[row][col].plot(valid_t[binned_spike_indexes], rec[binned_spike_indexes], 'k.', label='Spikes')
            axes[row][col].set_title('gid: {}'.format(gid), fontsize=mpl.rcParams['font.size'])
            axes[row][col].set_xlim(xlim)
        axes[0][cols-1].legend(loc='center left', frameon=False, framealpha=0.5, bbox_to_anchor=(1., 0.5),
                               fontsize=mpl.rcParams['font.size'])
        clean_axes(axes)
        fig.suptitle('Voltage recordings: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.85)
        fig.show()


def plot_connection_weights_heatmaps_from_matrix(connection_weights_matrix_dict, gids_sorted=True, pop_order=None,
                                                 label_dict=None):
    """

    :param connection_weights_matrix_dict: nested dict of 2d array of float
    :param binned_t_edges: array
    :param gids_sorted: bool; whether to label cell ids as sorted
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param normalize_t: bool
    """
    if pop_order is None:
        post_pop_order = sorted(list(connection_weights_matrix_dict.keys()))
        pre_pop_order = set()
        for post_pop in connection_weights_matrix_dict:
            for pre_pop in connection_weights_matrix_dict[post_pop]:
                pre_pop_order.add(pre_pop)
        pre_pop_order = sorted(list(pre_pop_order))
    else:
        post_pop_order = []
        pre_pop_order = []
        pre_pop_set = set()
        for post_pop in pop_order:
            if post_pop in connection_weights_matrix_dict:
                post_pop_order.append(post_pop)
                for pre_pop in connection_weights_matrix_dict[post_pop]:
                    pre_pop_set.add(pre_pop)
        for pre_pop in pop_order:
            if pre_pop in pre_pop_set:
                pre_pop_order.append(pre_pop)

    if label_dict is None:
        label_dict = {}
        for pop_name in post_pop_order + pre_pop_order:
            label_dict[pop_name] = pop_name

    fig, axes = plt.subplots(len(post_pop_order), len(pre_pop_order),
                             figsize=(4. * len(pre_pop_order), 3.5 * len(post_pop_order)))
    for j, post_pop_name in enumerate(post_pop_order):
        for i, pre_pop_name in enumerate(pre_pop_order):
            if pre_pop_name not in connection_weights_matrix_dict[post_pop_name]:
                continue
            this_connection_weight_matrix = connection_weights_matrix_dict[post_pop_name][pre_pop_name]
            extent = (-0.5, this_connection_weight_matrix.shape[1] - 0.5,
                      this_connection_weight_matrix.shape[0] - 0.5, -0.5)
            plot_heatmap_from_matrix(this_connection_weight_matrix, ax=axes[j,i], aspect='auto',
                                     cbar_label='Synaptic weight', extent=extent, vmin=0.,
                                     interpolation='none', cmap='binary')
            axes[j,i].set_title('%s <- %s' % (label_dict[post_pop_name], label_dict[pre_pop_name]),
                              fontsize=mpl.rcParams['font.size'])
            if gids_sorted:
                axes[j,i].set_xlabel('Sorted Cell ID\nPresynaptic (%s)' % label_dict[pre_pop_name])
            else:
                axes[j,i].set_xlabel('Cell ID\nPresynaptic(%s)' % label_dict[pre_pop_name])
            if i == 0:
                if gids_sorted:
                    axes[j,i].set_ylabel('Postsynaptic (%s)\nSorted Cell ID' % label_dict[post_pop_name])
                else:
                    axes[j,i].set_ylabel('Postsynaptic (%s)\nCell ID' % label_dict[post_pop_name])

        clean_axes(axes)
        fig.tight_layout()
        fig.show()


def plot_weight_matrix(connection_weights_dict, pop_gid_ranges, tuning_peak_locs=None, pop_names=None):
    """
    Plots heat maps of connection strengths across all connected cell populations. If input activity or input weights
    are spatially tuned, cell ids are also sorted by peak location.
    Assumes rows and columns of connection weight matrices are sorted by gid, not by peak location.
    :param connection_weights_dict: nested dict: {'target_pop_name': {'source_pop_name':
                2d array of float with shape (num_target_cells, num_source_cells) } }
    :param pop_gid_ranges: dict: {'pop_name', tuple of int}
    :param tuning_peak_locs: nested dict: {'pop_name': {'gid': float} }
    :param pop_names: list of str
    """
    if pop_names is None:
        pop_names = sorted(list(connection_weights_dict.keys()))
    sorted_gid_indexes = dict()
    for target_pop_name in pop_names:
        if target_pop_name not in connection_weights_dict:
            raise RuntimeError('plot_weight_matrix: missing population: %s' % target_pop_name)
        if tuning_peak_locs is not None and target_pop_name not in sorted_gid_indexes and \
                target_pop_name in tuning_peak_locs and len(tuning_peak_locs[target_pop_name]) > 0:
            this_ordered_peak_locs = \
                np.array([tuning_peak_locs[target_pop_name][gid] for gid in range(*pop_gid_ranges[target_pop_name])])
            sorted_gid_indexes[target_pop_name] = np.argsort(this_ordered_peak_locs)
        target_pop_size = pop_gid_ranges[target_pop_name][1] - pop_gid_ranges[target_pop_name][0]
        cols = len(connection_weights_dict[target_pop_name])
        fig, axes = plt.subplots(1, cols, sharey=True, figsize=(5 * cols, 5))
        y_interval = max(2, target_pop_size // 10)
        yticks = list(range(0, target_pop_size, y_interval))
        ylabels = np.add(yticks, pop_gid_ranges[target_pop_name][0])
        if target_pop_name in sorted_gid_indexes:
            axes[0].set_ylabel('Target: %s\nSorted Cell ID' % target_pop_name)
        else:
            axes[0].set_ylabel('Target: %s\nCell ID' % target_pop_name)

        for col, source_pop_name in enumerate(connection_weights_dict[target_pop_name]):
            if tuning_peak_locs is not None and source_pop_name not in sorted_gid_indexes and \
                    source_pop_name in tuning_peak_locs and len(tuning_peak_locs[source_pop_name]) > 0:
                this_ordered_peak_locs = \
                    np.array([tuning_peak_locs[source_pop_name][gid]
                              for gid in range(*pop_gid_ranges[source_pop_name])])
                sorted_gid_indexes[source_pop_name] = np.argsort(this_ordered_peak_locs)
            weight_matrix = np.copy(connection_weights_dict[target_pop_name][source_pop_name])
            if target_pop_name in sorted_gid_indexes:
                weight_matrix[:] = weight_matrix[sorted_gid_indexes[target_pop_name], :]
            if source_pop_name in sorted_gid_indexes:
                weight_matrix[:] = weight_matrix[:, sorted_gid_indexes[source_pop_name]]

            source_pop_size = pop_gid_ranges[source_pop_name][1] - pop_gid_ranges[source_pop_name][0]
            x_interval = max(2, source_pop_size // 10)
            xticks = list(range(0, source_pop_size, x_interval))
            xlabels = np.add(xticks, pop_gid_ranges[source_pop_name][0])
            if source_pop_name in sorted_gid_indexes:
                axes[col].set_xlabel('Sorted Cell ID\nSource: %s' % source_pop_name)
            else:
                axes[col].set_xlabel('Cell ID\nSource: %s' % source_pop_name)

            plot_heatmap_from_matrix(weight_matrix, xticks=xticks, xtick_labels=xlabels, yticks=yticks,
                                     ytick_labels=ylabels, ax=axes[col], aspect='auto', cbar_label='Synaptic weight',
                                     vmin=0.)
        clean_axes(axes)
        fig.suptitle('Connection weights onto %s population' % target_pop_name, fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, wspace=0.2)
        fig.show()


def plot_firing_rate_heatmaps(firing_rates_dict, input_t, valid_t=None, pop_names=None, tuning_peak_locs=None,
                              sorted_gids=None):
    """

    :param firing_rates_dict: dict of array
    :param input_t: array
    :param valid_t: array
    :param pop_names: list of str
    :param tuning_peak_locs: dict: {pop_name (str): {gid (int): float} }
    :param sorted_gids: dict: {pop_name (str): array of int}
    """
    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    if pop_names is None:
        pop_names = sorted(list(firing_rates_dict.keys()))
    for pop_name in pop_names:
        if tuning_peak_locs is not None:
            sort = pop_name in tuning_peak_locs and len(tuning_peak_locs[pop_name]) > 0
            if sort:
                sorted_indexes = np.argsort(list(tuning_peak_locs[pop_name].values()))
                this_sorted_gids = np.array(list(tuning_peak_locs[pop_name].keys()))[sorted_indexes]
            else:
                this_sorted_gids = sorted(list(firing_rates_dict[pop_name].keys()))
        elif sorted_gids is not None:
            sort = pop_name in sorted_gids and len(sorted_gids[pop_name]) > 0
            if sort:
                this_sorted_gids = sorted_gids[pop_name]
            else:
                this_sorted_gids = sorted(list(firing_rates_dict[pop_name].keys()))
        else:
            sort = False
            this_sorted_gids = sorted(list(firing_rates_dict[pop_name].keys()))
        fig, axes = plt.subplots()
        rate_matrix = np.empty((len(this_sorted_gids), len(valid_t)), dtype='float32')
        for i, gid in enumerate(this_sorted_gids):
            rate_matrix[i][:] = firing_rates_dict[pop_name][gid][valid_indexes]
        y_interval = max(2, len(this_sorted_gids) // 10)
        min_gid = np.min(this_sorted_gids)
        yticks = list(range(0, len(this_sorted_gids), y_interval))
        ylabels = np.add(yticks, min_gid)
        x_interval = max(1, math.floor(len(valid_t) / 10))
        xticks = list(range(0, len(valid_t), x_interval))
        xlabels = np.array(valid_t)[xticks].astype('int32')
        plot_heatmap_from_matrix(rate_matrix, xticks=xticks, xtick_labels=xlabels, yticks=yticks,
                                 ytick_labels=ylabels, ax=axes, aspect='auto', cbar_label='Firing rate (Hz)',
                                 vmin=0.)
        axes.set_xlabel('Time (ms)')
        if sort:
            axes.set_title('Firing rate: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
            axes.set_ylabel('Sorted Cell ID')
        else:
            axes.set_title('Firing rate: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
            axes.set_ylabel('Cell ID')
        clean_axes(axes)
        fig.tight_layout()
        fig.show()


def plot_firing_rate_heatmaps_from_matrix(firing_rates_matrix_dict, binned_t_edges, gids_sorted=True, pop_order=None,
                                          label_dict=None, normalize_t=True):
    """

    :param firing_rates_matrix_dict: dict of array
    :param binned_t_edges: array
    :param gids_sorted: bool; whether to label cell ids as sorted
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param normalize_t: bool
    """
    if pop_order is None:
        pop_order = sorted(list(firing_rates_matrix_dict.keys()))

    num_pops = len(firing_rates_matrix_dict)
    fig, axes = plt.subplots(1, num_pops, figsize=(4. * num_pops, 3.5))

    for i, pop_name in enumerate(pop_order):
        this_firing_rate_matrix = firing_rates_matrix_dict[pop_name]
        if normalize_t:
            extent = (0., 1., len(this_firing_rate_matrix) - 0.5, -0.5)
        else:
            extent = (binned_t_edges[0] / 1000., binned_t_edges[-1] / 1000., len(this_firing_rate_matrix) - 0.5, -0.5)

        plot_heatmap_from_matrix(this_firing_rate_matrix, ax=axes[i], aspect='auto', cbar_label='Firing rate (Hz)',
                                 vmin=0., extent=extent)
        if normalize_t:
            axes[i].set_xlabel('Normalized position')
        else:
            axes[i].set_xlabel('Time (s)')
        if gids_sorted:
            axes[i].set_ylabel('Sorted Cell ID')
        else:
            axes[i].set_ylabel('Cell ID')
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        axes[i].set_title('%s' % label, fontsize=mpl.rcParams['font.size'])

    clean_axes(axes)
    fig.tight_layout()
    fig.show()


def plot_average_selectivity(binned_t_edges, centered_firing_rate_mean_dict, centered_firing_rate_sem_dict=None,
                             pop_order=None, label_dict=None, color_dict=None):
    """
    Plot the average selectivity (firing rates centered around the peak of their activity) for each cell population.
    :param binned_t_edges: array of float
    :param centered_firing_rate_mean_dict: dict of array of float
    :param centered_firing_rate_sem_dict: dict of array of float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    """
    center_index = len(binned_t_edges) // 2
    offset_binned_t_edges = (binned_t_edges - binned_t_edges[center_index]) / 1000.

    fig, axis = plt.subplots(1, figsize=(3.5, 3.5))

    if pop_order is None:
        pop_order = sorted(list(centered_firing_rate_mean_dict.keys()))

    for pop_name in pop_order:
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        if color_dict is not None:
            color = color_dict[pop_name]
            axis.plot(offset_binned_t_edges, centered_firing_rate_mean_dict[pop_name], color=color, label=label)
            if centered_firing_rate_sem_dict is not None:
                axis.fill_between(offset_binned_t_edges,
                                  centered_firing_rate_mean_dict[pop_name] - centered_firing_rate_sem_dict[pop_name],
                                  centered_firing_rate_mean_dict[pop_name] + centered_firing_rate_sem_dict[pop_name],
                                  color=color, alpha=0.25, linewidth=0)
        else:
            axis.plot(offset_binned_t_edges, centered_firing_rate_mean_dict[pop_name], label=label)
            if centered_firing_rate_sem_dict is not None:
                axis.fill_between(offset_binned_t_edges,
                                  centered_firing_rate_mean_dict[pop_name] - centered_firing_rate_sem_dict[pop_name],
                                  centered_firing_rate_mean_dict[pop_name] + centered_firing_rate_sem_dict[pop_name],
                                  alpha=0.25, linewidth=0)

    axis.set_ylim((0., axis.get_ylim()[1]))
    axis.set_xlim((offset_binned_t_edges[0], offset_binned_t_edges[-1]))
    axis.set_title('Selectivity', fontsize=mpl.rcParams['font.size'])
    axis.set_ylabel('Firing rate (Hz)')
    axis.set_xlabel('Time to peak firing rate (s)')
    axis.legend(loc='best', fontsize=mpl.rcParams['font.size'], frameon=False, framealpha=0.5, handlelength=1)
    clean_axes([axis])
    fig.tight_layout()
    fig.show()


def plot_replay_spike_rasters(spike_times_dict, binned_t_edges, sorted_gid_dict, trial_key=None, pop_order=None,
                              label_dict=None):
    """

    :param spike_times_dict: nested dict: {pop_name (str): {gid (int): array of float) } }
    :param binned_t_edges: array of float (ms)
    :param sorted_gid_dict: dict of array of int
    :param trial_key: str or int
    :param pop_order: list of str
    :param label_dict: dict of str
    """
    if pop_order is None:
        pop_order = sorted(list(spike_times_dict.keys()))

    fig, axes = plt.subplots(1, len(pop_order), figsize=(2.2 * len(pop_order), 2.4))
    this_cmap = copy.copy(plt.get_cmap('binary'))
    this_cmap.set_bad(this_cmap(0.))
    for col, pop_name in enumerate(pop_order):
        for i, gid in enumerate(sorted_gid_dict[pop_name]):
            this_spike_times = spike_times_dict[pop_name][gid]
            axes[col].scatter(this_spike_times, np.ones_like(this_spike_times) * i + 0.5, c='k', s=0.01,
                                 rasterized=True)
        axes[col].set_xlabel('Time (ms)')
        axes[col].set_ylim((len(sorted_gid_dict[pop_name]), 0))
        axes[col].set_xlim((binned_t_edges[0], binned_t_edges[-1]))
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        axes[col].set_title(label, fontsize=mpl.rcParams['font.size'])
    axes[0].set_ylabel('Sorted Cell ID')
    if trial_key is not None:
        fig.suptitle('Trial # %s' % trial_key, y=0.99, fontsize=mpl.rcParams['font.size'])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5, hspace=0.4, top=0.8)


def visualize_connections(pop_gid_ranges, pop_cell_types, pop_syn_proportions, pop_cell_positions, connectivity_dict, n=1,
                          plot_from_hdf5=True):
    """
    :param pop_cell_positions: nested dict
    :param connectivity_dict: nested dict
    :param n: int
    """
    for target_pop_name in pop_syn_proportions:
        if pop_cell_types[target_pop_name] == 'input':
            continue
        start_idx, end_idx = pop_gid_ranges[target_pop_name]
        target_gids = random.sample(range(start_idx, end_idx), n)
        for target_gid in target_gids:
            if plot_from_hdf5: target_gid = str(target_gid)
            target_loc = pop_cell_positions[target_pop_name][target_gid]
            for syn_type in pop_syn_proportions[target_pop_name]:
                for source_pop_name in pop_syn_proportions[target_pop_name][syn_type]:
                    source_gids = connectivity_dict[target_pop_name][target_gid][source_pop_name]
                    if not len(source_gids):  # if the list is empty
                        continue
                    xs = []
                    ys = []
                    for source_gid in source_gids:
                        if plot_from_hdf5: source_gid = str(source_gid)
                        xs.append(pop_cell_positions[source_pop_name][source_gid][0])
                        ys.append(pop_cell_positions[source_pop_name][source_gid][1])
                    vals, xedge, yedge = np.histogram2d(x=xs, y=ys, bins=np.linspace(-1.0, 1.0, 51))
                    fig = plt.figure()
                    plt.pcolor(xedge, yedge, vals)
                    plt.title("Cell {} at {}, {} to {} via {} syn".format(target_gid, target_loc, source_pop_name,
                                                                          target_pop_name, syn_type),
                              fontsize=mpl.rcParams['font.size'])
                    fig.show()


def plot_2D_connection_distance(pop_syn_proportions, pop_cell_positions, connectivity_dict):
    """
    Generate 2D histograms of relative distances
    :param pop_syn_proportions: nested dict
    :param pop_cell_positions: nested dict
    :param connectivity_dict: nested dict
    """
    for target_pop_name in pop_syn_proportions:
        if len(pop_cell_positions) == 0 or len(next(iter(viewvalues(pop_cell_positions[target_pop_name])))) < 2:
            print('plot_2D_connection_distance: cell position data is absent or spatial dimension < 2')
            sys.stdout.fluish()
            continue
        for syn_type in pop_syn_proportions[target_pop_name]:
            for source_pop_name in pop_syn_proportions[target_pop_name][syn_type]:
                x_dist = []
                y_dist = []
                for target_gid in connectivity_dict[target_pop_name]:
                    x_target = pop_cell_positions[target_pop_name][target_gid][0]
                    y_target = pop_cell_positions[target_pop_name][target_gid][1]
                    if source_pop_name in connectivity_dict[target_pop_name][target_gid]:
                        for source_gid in connectivity_dict[target_pop_name][target_gid][source_pop_name]:
                            x_source = pop_cell_positions[source_pop_name][source_gid][0]
                            y_source = pop_cell_positions[source_pop_name][source_gid][1]
                            x_dist.append(x_source - x_target)
                            y_dist.append(y_source - y_target)
                fig = plt.figure()
                plt.hist2d(x_dist, y_dist)
                plt.colorbar().set_label("Count")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title("{} to {} distances".format(source_pop_name, target_pop_name),
                          fontsize=mpl.rcParams['font.size'])
                fig.show()


def get_firing_rate_matrix_dict_from_nested_dict(firing_rates_dict, binned_t_edges):
    """
    Given a dictionary of firing rates organized by population and gid, return a dictionary containing a firing rate
    matrix for each population. Also return a dictionary with the unsorted gid order of each matrix.
    :param firing_rates_dict: nested dict of float: {pop_name (str): {gid (int): float } }
    :param binned_t_edges: array of float
    :return: tuple: dict of 2d array of float: {pop_name: 2d array with shape (num_cells, num bins)},
                    dict of array of int: {pop_name: array of gids}
    """
    firing_rate_matrix_dict = {}
    gid_order_dict = {}

    for pop_name in firing_rates_dict:
        firing_rate_matrix_dict[pop_name] = np.empty((len(firing_rates_dict[pop_name]), len(binned_t_edges)))
        gid_order = sorted(list(firing_rates_dict[pop_name].keys()))
        for i, gid in enumerate(gid_order):
            firing_rate_matrix_dict[pop_name][i, :] = firing_rates_dict[pop_name][gid]
        gid_order_dict[pop_name] = np.array(gid_order)

    return firing_rate_matrix_dict, gid_order_dict


def analyze_selectivity_from_firing_rate_matrix_dict(firing_rate_matrix_dict, gid_order_dict):
    """
    Given a dict of firing rates for each population, center the firing rates of each cell around the peak of its
    activity. Average across cells within a population. Return the mean and sem traces for each population, as well
    as a dict with an array of gids in order of peak activity.
    :param firing_rate_matrix_dict: dict of 2d array of float: {pop_name: 2d array with shape (num_cells, num bins)}
    :param gid_order_dict: dict of array of int
    :return: tuple: dict of array of float: {pop_name: array of length (num bins)}; population mean, peak centered,
                    dict of array of float: {pop_name: array of length (num bins)}; population sem, peak centered,
                    dict of array of int: {pop_name: array of gids}; sorted gid order
    """
    centered_firing_rate_mean_dict = {}
    centered_firing_rate_sem_dict = {}
    sorted_gid_dict = {}

    for pop_name in firing_rate_matrix_dict:
        centered_firing_rate_matrix = np.empty_like(firing_rate_matrix_dict[pop_name])
        peak_indexes = []
        center_index = centered_firing_rate_matrix.shape[1] // 2
        for i in range(len(centered_firing_rate_matrix)):
            rate = firing_rate_matrix_dict[pop_name][i]
            peak_index = np.argmax(rate)
            peak_indexes.append(peak_index)
            centered_firing_rate_matrix[i] = np.roll(rate, -peak_index + center_index)
        sorted_gid_indexes = np.argsort(peak_indexes)
        sorted_gid_dict[pop_name] = gid_order_dict[pop_name][sorted_gid_indexes]
        centered_firing_rate_mean_dict[pop_name] = np.mean(centered_firing_rate_matrix, axis=0)
        centered_firing_rate_sem_dict[pop_name] = \
            np.std(centered_firing_rate_matrix, axis=0) / np.sqrt(len(centered_firing_rate_matrix))

    return centered_firing_rate_mean_dict, centered_firing_rate_sem_dict, sorted_gid_dict


def sort_firing_rate_matrix_dict(firing_rate_matrix_dict, gid_order_dict, sorted_gid_dict):
    """
    Given a dictionary of firing rate matrices organized by population, re-sort the matrices according to the
    specified sorted_gid_dict.
    :param firing_rate_matrix_dict: dict of 2d array of float: {pop_name: 2d array with shape (num_cells, num bins)}
    :param gid_order_dict: dict of array of int; initial sorting
    :param sorted_gid_dict: dict of array of int; new sorting
    :return: dict of 2d array of float: {pop_name: 2d array with shape (num_cells, num bins)}
    """
    sorted_firing_rate_matrix_dict = {}
    for pop_name in firing_rate_matrix_dict:
        gid_index_dict = dict(zip(gid_order_dict[pop_name], range(len(gid_order_dict[pop_name]))))
        sorted_indexes = []
        for gid in sorted_gid_dict[pop_name]:
            index = gid_index_dict[gid]
            sorted_indexes.append(index)
        sorted_indexes = np.asarray(sorted_indexes)
        sorted_firing_rate_matrix_dict[pop_name] = firing_rate_matrix_dict[pop_name][sorted_indexes]

    return sorted_firing_rate_matrix_dict


def sort_connection_weights_matrix_dict(connection_weights_matrix_dict, gid_order_dict, sorted_gid_dict):
    """
    Given a dictionary of connection weight matrices organized by post-synaptic and pre-synaptic populations, re-sort
    the matrices according to the specified sorted_gid_dict.
    :param connection_weights_matrix_dict: dict of 2d array of float: {pop_name: 2d array with shape (num_cells, num bins)}
    :param gid_order_dict: dict of array of int; initial sorting
    :param sorted_gid_dict: dict of array of int; new sorting
    :return: nested dict of 2d array of float:
                {post_pop_name:
                    {pre_pop_name: 2d array with shape (num cells in post_pop, num cells in pre_pop)
                    }
                {
    """
    sorted_connection_weights_matrix_dict = {}
    sorted_indexes_dict = {}
    for pop_name in gid_order_dict:
        gid_index_dict = dict(zip(gid_order_dict[pop_name], range(len(gid_order_dict[pop_name]))))
        sorted_indexes = []
        for gid in sorted_gid_dict[pop_name]:
            index = gid_index_dict[gid]
            sorted_indexes.append(index)
        sorted_indexes_dict[pop_name] = np.asarray(sorted_indexes)
    for post_pop_name in connection_weights_matrix_dict:
        sorted_connection_weights_matrix_dict[post_pop_name] = {}
        for pre_pop_name in connection_weights_matrix_dict[post_pop_name]:
            sorted_connection_weights_matrix = \
                connection_weights_matrix_dict[post_pop_name][pre_pop_name][sorted_indexes_dict[post_pop_name], :]
            sorted_connection_weights_matrix_dict[post_pop_name][pre_pop_name] = \
                sorted_connection_weights_matrix[:, sorted_indexes_dict[pre_pop_name]]

    return sorted_connection_weights_matrix_dict


def get_trial_averaged_fft_power(fft_f, fft_power_dict_list):
    """
    Given lists of dicts containing data across trials or network instances, return dicts containing the mean and sem
    for fft power for each cell population.
    :param fft_f: array of float
    :param fft_power_dict_list: list of dict: {pop_name: array of float}
    :return: tuple of dict
    """
    fft_power_mean_dict = {}
    fft_power_sem_dict = {}
    num_instances = len(fft_power_dict_list)
    for fft_power_dict in fft_power_dict_list:
        for pop_name in fft_power_dict:
            if pop_name not in fft_power_mean_dict:
                fft_power_mean_dict[pop_name] = []
            fft_power_mean_dict[pop_name].append(fft_power_dict[pop_name])
    for pop_name in fft_power_mean_dict:
        fft_power_sem_dict[pop_name] = np.std(fft_power_mean_dict[pop_name], axis=0) / np.sqrt(num_instances)
        fft_power_mean_dict[pop_name] = np.mean(fft_power_mean_dict[pop_name], axis=0)

    return fft_power_mean_dict, fft_power_sem_dict


def analyze_spatial_modulation_across_instances(modulation_depth_dict_list, delta_peak_locs_dict_list):
    """

    :param modulation_depth_dict_list: list of dict {'predicted;actual': {pop_name: array}}
    :param delta_peak_locs_dict_list: dict: {pop_name: array}
    :return: tuple of dict
    """
    modulation_depth_instances_dict = {'predicted': {}, 'actual': {}}
    delta_peak_locs_instances_dict = {}
    for modulation_depth_instance in modulation_depth_dict_list:
        for condition in modulation_depth_instances_dict.keys():
            for pop_name in modulation_depth_instance[condition]:
                if pop_name not in modulation_depth_instances_dict[condition]:
                    modulation_depth_instances_dict[condition][pop_name] = []
                this_mean_modulation_depth = np.mean(modulation_depth_instance[condition][pop_name])
                modulation_depth_instances_dict[condition][pop_name].append(this_mean_modulation_depth)
    for delta_peak_locs_instance in delta_peak_locs_dict_list:
        for pop_name in delta_peak_locs_instance:
            if pop_name not in delta_peak_locs_instances_dict:
                delta_peak_locs_instances_dict[pop_name] = []
            this_mean_delta_peak_locs = np.mean(delta_peak_locs_instance[pop_name])
            delta_peak_locs_instances_dict[pop_name].append(this_mean_delta_peak_locs)

    return modulation_depth_instances_dict, delta_peak_locs_instances_dict


def plot_rhythmicity_psd(fft_f, fft_power_mean_dict, fft_power_sem_dict=None, freq_max=250., pop_order=None,
                         label_dict=None, color_dict=None, compressed_plot_format=False, title='Rhythmicity'):
    """

    :param fft_f: array of float
    :param fft_power_mean_dict: dict: {pop_name: array of float}
    :param fft_power_sem_dict: dict: {pop_name: array of float}
    :param freq_max: float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    :param compressed_plot_format: bool
    """
    if compressed_plot_format:
        fig, axes = plt.subplots(1, figsize=(3.2, 2.8))
    else:
        fig, axes = plt.subplots(1, figsize=(4., 3.5))
    freq_max_index = np.where(fft_f >= freq_max)[0][0]

    if pop_order is None:
        pop_order = sorted(list(fft_power_mean_dict.keys()))

    for pop_name in pop_order:
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        if color_dict is not None:
            color = color_dict[pop_name]
            axes.semilogy(fft_f[1:freq_max_index], fft_power_mean_dict[pop_name][1:freq_max_index],
                          label=label, color=color)
            if fft_power_sem_dict is not None:
                axes.fill_between(fft_f[1:freq_max_index], fft_power_mean_dict[pop_name][1:freq_max_index] +
                                  fft_power_sem_dict[pop_name][1:freq_max_index],
                                  fft_power_mean_dict[pop_name][1:freq_max_index] -
                                  fft_power_sem_dict[pop_name][1:freq_max_index], alpha=0.25, linewidth=0,
                                  color=color)
        else:
            axes.semilogy(fft_f[1:freq_max_index], fft_power_mean_dict[pop_name][1:freq_max_index],
                          label=label)
            if fft_power_sem_dict is not None:
                axes.fill_between(fft_f[1:freq_max_index], fft_power_mean_dict[pop_name][1:freq_max_index] +
                                  fft_power_sem_dict[pop_name][1:freq_max_index],
                                  fft_power_mean_dict[pop_name][1:freq_max_index] -
                                  fft_power_sem_dict[pop_name][1:freq_max_index], alpha=0.25, linewidth=0)
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Power spectral density\n(Hz$^{2}$/Hz) (Log scale)')
    axes.set_xlim((0., fft_f[freq_max_index]))
    axes.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes.set_title(title, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    fig.tight_layout()
    fig.show()


def get_trial_averaged_firing_rate_matrix_dict(firing_rate_matrix_dict_list):
    """
    Given a list of firing rate matrix dicts with the same default gid ordering, average across the trials in the list
    and returned the mean firing rate matrix dict.
    :param firing_rate_matrix_dict_list: list of dict of 2d array of float (num cells, num time points)
    :return: list of dict of 2d array of float (num cells, num time points)
    """
    mean_firing_rate_matrix_dict = dict()
    for pop_name in firing_rate_matrix_dict_list[0]:
        mean_firing_rate_matrix_dict[pop_name] = \
            np.mean([this_firing_rate_matrix_dict[pop_name]
                     for this_firing_rate_matrix_dict in firing_rate_matrix_dict_list], axis=0)

    return mean_firing_rate_matrix_dict


def get_pop_bandpass_envelope_fft(filter_envelope_dict, dt):
    """
    Given the envelope of a bandpass-filtered signal, determine the frequency modulation. Return a power spectral
    density.
    :param filter_envelope_dict: dict {'pop_name': array}
    :param dt: float
    :return: tuple of dict {'pop_name': array}
    """
    sampling_rate = 1000. / dt
    fft_f_dict, fft_power_dict = {}, {}
    for pop_name in filter_envelope_dict:
        fft_f_dict[pop_name], fft_power_dict[pop_name] = periodogram(filter_envelope_dict[pop_name], fs=sampling_rate)

    return fft_f_dict, fft_power_dict


def get_modulation_depth(signal):
    ceil_indexes = np.where(signal >= np.percentile(signal, 90))
    ceil = np.nanmean(signal[ceil_indexes])
    mean_signal = np.nanmean(signal)
    return ceil / mean_signal


def analyze_selectivity_input_output(binned_t_edges, firing_rate_matrix_dict, pop_gid_ranges, input_populations,
                                     output_populations, connectivity_dict, connection_weights_dict):
    """

    :param binned_t_edges:
    :param firing_rate_matrix_dict:
    :param pop_gid_ranges:
    :param input_populations:
    :param output_populations:
    :param connectivity_dict:
    :param connection_weights_dict:
    :return:
    """
    track_length = binned_t_edges[-1]
    peak_locs = defaultdict(lambda: defaultdict(list))
    delta_peak_locs = {}
    modulation_depth = {'actual': {}, 'predicted': {}}
    mean_delta_peak_locs = {}
    mean_modulation_depth = {'actual': {}, 'predicted': {}}
    for post_pop in output_populations:
        modulation_depth['actual'][post_pop] = []
        modulation_depth['predicted'][post_pop] = []
        delta_peak_locs[post_pop] = []
        post_start_gid = pop_gid_ranges[post_pop][0]
        for post_gid in range(*pop_gid_ranges[post_pop]):
            predicted = np.zeros_like(binned_t_edges)
            for pre_pop in input_populations:
                pre_start_gid = pop_gid_ranges[pre_pop][0]
                if pre_pop in connectivity_dict[post_pop][post_gid]:
                    for pre_gid in connectivity_dict[post_pop][post_gid][pre_pop]:
                        this_rate = firing_rate_matrix_dict[pre_pop][pre_gid - pre_start_gid]
                        this_weight = \
                            connection_weights_dict[post_pop][pre_pop][post_gid - post_start_gid][pre_gid - pre_start_gid]
                        predicted += this_rate * this_weight
            predicted_peak_index = np.argmax(predicted)
            peak_locs['predicted'][post_pop].append(binned_t_edges[predicted_peak_index])
            modulation_depth['predicted'][post_pop].append(get_modulation_depth(predicted))
            actual_rate = firing_rate_matrix_dict[post_pop][post_gid - post_start_gid]
            actual_peak_index = np.argmax(actual_rate)
            peak_locs['actual'][post_pop].append(binned_t_edges[actual_peak_index])
            this_modulation_depth = get_modulation_depth(actual_rate)
            if not np.isnan(this_modulation_depth):
                modulation_depth['actual'][post_pop].append(this_modulation_depth)
            this_delta_peak_loc = np.abs(binned_t_edges[actual_peak_index] - binned_t_edges[predicted_peak_index])
            this_delta_peak_loc /= track_length
            if this_delta_peak_loc > 0.5:
                this_delta_peak_loc = 1. - this_delta_peak_loc
            delta_peak_locs[post_pop].append(this_delta_peak_loc)
        mean_delta_peak_locs[post_pop] = np.mean(delta_peak_locs[post_pop])
        mean_modulation_depth['predicted'][post_pop] = np.mean(modulation_depth['predicted'][post_pop])
        mean_modulation_depth['actual'][post_pop] = np.mean(modulation_depth['actual'][post_pop])

    return modulation_depth, delta_peak_locs


def plot_selectivity_input_output(modulation_depth_dict, delta_peak_locs_dict, pop_order=None, color_dict=None,
                                  label_dict=None):
    """

    :param modulation_depth_dict:
    :param delta_peak_locs_dict:
    :param pop_order:
    :param color_dict:
    :param label_dict:
    """
    from matplotlib.lines import Line2D

    if pop_order is None:
        pop_order = sorted(list(delta_peak_locs_dict.keys()))
    if label_dict is None:
        label_dict = {}
        for pop_name in pop_order:
            label_dict[pop_name] = pop_name
    if color_dict is None:
        color_dict = {}
        for pop_name in pop_order:
            color_dict[pop_name] = None

    fig, axes = plt.subplots(1, 2, figsize=(8., 3.5))
    mod_depth_plot_labels = []
    lines = []
    handles = []
    pos_start = 0
    for pop_name in pop_order:
        lines.append(Line2D([0], [0], color=color_dict[pop_name]))
        handles.append(label_dict[pop_name])
        mod_depth_plot_items = []
        for key, label in zip(['predicted', 'actual'], ['Expected', 'Actual']):
            mod_depth_plot_items.append(modulation_depth_dict[key][pop_name])
            mod_depth_plot_labels.append(label)
        bp = axes[0].boxplot(mod_depth_plot_items, positions=[pos_start, pos_start + 1], patch_artist=True,
                             showfliers=False)
        if color_dict[pop_name] is not None:
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=color_dict[pop_name])
        for patch in bp['boxes']:
            patch.set(facecolor='white')
        pos_start += 2

    axes[0].set_xticks(range(len(pop_order) * 2))
    axes[0].set_xticklabels(mod_depth_plot_labels, fontsize=mpl.rcParams['font.size'] - 4)
    axes[0].set_ylabel('Spatial modulation')
    axes[0].set_ylim((min(axes[0].get_ylim()[0], 1.), axes[0].get_ylim()[1]))
    axes[0].legend(lines, handles, loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'],
                   handlelength=1)

    delta_peak_plot_labels = []
    pos_start = 0
    for pop_name in pop_order:
        for key, label in zip(['predicted', 'actual'], ['Expected', 'Actual']):
            mod_depth_plot_items.append(modulation_depth_dict[key][pop_name])
            mod_depth_plot_labels.append(label)
        delta_peak_plot_labels.append(label_dict[pop_name])
        bp = axes[1].boxplot([delta_peak_locs_dict[pop_name]], positions=[pos_start], patch_artist=True,
                             showfliers=False)
        if color_dict[pop_name] is not None:
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=color_dict[pop_name])
        for patch in bp['boxes']:
            patch.set(facecolor='white')
        pos_start += 1

    axes[1].set_xticks(range(len(pop_order)))
    axes[1].set_xticklabels(delta_peak_plot_labels)
    axes[1].set_ylabel('Difference in\npeak locations\n|Actual - Expected|')
    axes[1].set_ylim((min(axes[1].get_ylim()[0], 0.), max(axes[1].get_ylim()[1], 0.1)))
    clean_axes(axes)
    fig.tight_layout()
    fig.show()


def analyze_simple_network_run_data_from_file(data_file_path, data_key='0', example_trial=0, fine_binned_dt=1.,
                                              coarse_binned_dt=20., filter_order=None, filter_label_dict=None,
                                              filter_color_dict=None, filter_xlim_dict=None, pop_order=None,
                                              label_dict=None, color_dict=None, plot=True, verbose=False):
    """
    Given an hdf5 file containing multiple trials from one network instance, import data, process data, plot example
    traces from the specified trial, and plot trial averages for sparsity, selectivity, and rhythmicity.
    :param data_file_path: str (path)
    :param data_key: int or str
    :param example_trial: int
    :param fine_binned_dt: float
    :param coarse_binned_dt: float
    :param filter_order: list of str
    :param filter_label_dict: dict
    :param filter_color_dict: dict
    :param filter_xlim_dict: dict of tuple of float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    :param plot: bool
    :param verbose: bool
    """
    if not os.path.isfile(data_file_path):
        raise IOError('analyze_simple_network_run_data_from_file: invalid data file path: %s' % data_file_path)

    trial_key_list = []
    full_spike_times_dict_list = []
    filter_bands = dict()
    subset_full_voltage_rec_dict = defaultdict(dict)
    connection_weights_matrix_dict = dict()
    tuning_peak_locs = dict()
    connectivity_dict = dict()
    pop_syn_proportions = dict()
    pop_cell_positions = dict()

    group_key = 'simple_network_exported_run_data'
    shared_context_key = 'shared_context'
    with h5py.File(data_file_path, 'r') as f:
        group = get_h5py_group(f, [data_key, group_key])
        subgroup = group[shared_context_key]
        connectivity_type = get_h5py_attr(subgroup.attrs, 'connectivity_type')
        active_rate_threshold = subgroup.attrs['active_rate_threshold']
        duration = get_h5py_attr(subgroup.attrs, 'duration')
        buffer = get_h5py_attr(subgroup.attrs, 'buffer')
        network_id = get_h5py_attr(subgroup.attrs, 'network_id')
        network_instance = get_h5py_attr(subgroup.attrs, 'network_instance')
        baks_alpha = get_h5py_attr(subgroup.attrs, 'baks_alpha')
        baks_beta = get_h5py_attr(subgroup.attrs, 'baks_beta')
        baks_pad_dur = get_h5py_attr(subgroup.attrs, 'baks_pad_dur')
        baks_wrap_around = get_h5py_attr(subgroup.attrs, 'baks_wrap_around')

        if 'full_rec_t' in subgroup:
            full_rec_t = subgroup['full_rec_t'][:]
            dt = full_rec_t[1] - full_rec_t[0]
        else:
            full_rec_t = None
            dt = None

        pop_gid_ranges = dict()
        for pop_name in subgroup['pop_gid_ranges']:
            pop_gid_ranges[pop_name] = tuple(subgroup['pop_gid_ranges'][pop_name][:])
        data_group = subgroup['filter_bands']
        for this_filter in data_group:
            filter_bands[this_filter] = data_group[this_filter][:]
        data_group = subgroup['connection_weights']
        for target_pop_name in data_group:
            connection_weights_matrix_dict[target_pop_name] = dict()
            for source_pop_name in data_group[target_pop_name]:
                connection_weights_matrix_dict[target_pop_name][source_pop_name] = \
                    data_group[target_pop_name][source_pop_name][:]
        if 'tuning_peak_locs' in subgroup and len(subgroup['tuning_peak_locs']) > 0:
            data_group = subgroup['tuning_peak_locs']
            for pop_name in data_group:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(data_group[pop_name]['target_gids'], data_group[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        data_group = subgroup['connectivity']
        for target_pop_name in data_group:
            connectivity_dict[target_pop_name] = dict()
            for target_gid_key in data_group[target_pop_name]:
                target_gid = int(target_gid_key)
                connectivity_dict[target_pop_name][target_gid] = dict()
                for source_pop_name in data_group[target_pop_name][target_gid_key]:
                    connectivity_dict[target_pop_name][target_gid][source_pop_name] = \
                        data_group[target_pop_name][target_gid_key][source_pop_name][:]
        data_group = subgroup['pop_syn_proportions']
        for target_pop_name in data_group:
            pop_syn_proportions[target_pop_name] = dict()
            for syn_type in data_group[target_pop_name]:
                pop_syn_proportions[target_pop_name][syn_type] = dict()
                source_pop_names = data_group[target_pop_name][syn_type]['source_pop_names'][:].astype('str')
                for source_pop_name, syn_proportion in zip(source_pop_names,
                                                           data_group[target_pop_name][syn_type]['syn_proportions'][:]):
                    pop_syn_proportions[target_pop_name][syn_type][source_pop_name] = syn_proportion
        data_group = subgroup['pop_cell_positions']
        for pop_name in data_group:
            pop_cell_positions[pop_name] = dict()
            for gid, position in zip(data_group[pop_name]['gids'][:], data_group[pop_name]['positions'][:]):
                pop_cell_positions[pop_name][gid] = position

        example_trial_key = str(example_trial)
        for trial_key in (key for key in group if key != shared_context_key):
            subgroup = group[trial_key]
            data_group = subgroup['full_spike_times']
            full_spike_times_dict = defaultdict(dict)
            for pop_name in data_group:
                for gid_key in data_group[pop_name]:
                    full_spike_times_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]
            full_spike_times_dict_list.append(full_spike_times_dict)
            trial_key_list.append(trial_key)
            if example_trial is not None and trial_key == example_trial_key:
                equilibrate = get_h5py_attr(subgroup.attrs, 'equilibrate')
                if 'subset_full_voltage_recs' in subgroup:
                    data_group = subgroup['subset_full_voltage_recs']
                    for pop_name in data_group:
                        for gid_key in data_group[pop_name]:
                            subset_full_voltage_rec_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]
                else:
                    subset_full_voltage_rec_dict = None

    buffered_binned_t_edges = \
        np.arange(-buffer, duration + buffer + fine_binned_dt / 2., fine_binned_dt)
    buffered_binned_t = buffered_binned_t_edges[:-1] + fine_binned_dt / 2.
    fine_binned_t_edges = np.arange(0., duration + fine_binned_dt / 2., fine_binned_dt)
    fine_binned_t = fine_binned_t_edges[:-1] + fine_binned_dt / 2.
    binned_t_edges = np.arange(0., duration + coarse_binned_dt / 2., coarse_binned_dt)
    binned_t = binned_t_edges[:-1] + coarse_binned_dt / 2.

    firing_rate_matrix_dict_list = []
    mean_rate_active_cells_dict_list = []
    pop_fraction_active_dict_list = []
    fft_power_dict_list = []
    fft_power_nested_gamma_dict_list = []

    for trial_key, full_spike_times_dict in zip(trial_key_list, full_spike_times_dict_list):
        current_time = time.time()
        if example_trial is not None and trial_key == example_trial_key:
            this_plot = plot
        else:
            this_plot = False
        binned_firing_rates_dict = \
            infer_firing_rates_baks(full_spike_times_dict, binned_t_edges, alpha=baks_alpha, beta=baks_beta,
                                    pad_dur=baks_pad_dur, wrap_around=baks_wrap_around)
        firing_rate_matrix_dict, gid_order_dict = \
            get_firing_rate_matrix_dict_from_nested_dict(binned_firing_rates_dict, binned_t_edges)
        firing_rate_matrix_dict_list.append(firing_rate_matrix_dict)

        buffered_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, buffered_binned_t_edges)
        buffered_pop_mean_rate_from_binned_spike_count_dict = \
            get_pop_mean_rate_from_binned_spike_count(buffered_binned_spike_count_dict, dt=fine_binned_dt)

        binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, binned_t_edges)
        firing_rates_from_binned_spike_count_dict = \
            get_firing_rates_from_binned_spike_count_dict(binned_spike_count_dict, bin_dur=coarse_binned_dt,
                                                          smooth=150., wrap=True)

        mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict = \
            get_pop_activity_stats(firing_rates_from_binned_spike_count_dict, input_t=binned_t,
                                   threshold=active_rate_threshold)

        mean_rate_active_cells_dict_list.append(mean_rate_active_cells_dict)
        pop_fraction_active_dict_list.append(pop_fraction_active_dict)

        fft_f_dict, fft_power_dict, filter_psd_f_dict, filter_psd_power_dict, filter_envelope_dict, \
        filter_envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict = \
            get_pop_bandpass_filtered_signal_stats(buffered_pop_mean_rate_from_binned_spike_count_dict,
                                                   filter_bands, input_t=buffered_binned_t,
                                                   valid_t=buffered_binned_t, output_t=fine_binned_t, pad=True,
                                                   filter_order=filter_order, filter_label_dict=filter_label_dict,
                                                   filter_color_dict=filter_color_dict,
                                                   filter_xlim_dict=filter_xlim_dict, pop_order=pop_order,
                                                   label_dict=label_dict, color_dict=color_dict, plot=this_plot)
        fft_power_dict_list.append(fft_power_dict)

        if 'Gamma' in filter_envelope_dict:
            fft_f_nested_gamma_dict, fft_power_nested_gamma_dict = \
                get_pop_bandpass_envelope_fft(filter_envelope_dict['Gamma'], dt=fine_binned_dt)
            fft_power_nested_gamma_dict_list.append(fft_power_nested_gamma_dict)

        if this_plot:
            plot_compare_binned_spike_counts(buffered_binned_spike_count_dict, buffered_binned_t, xlim=(0., 1000.),
                                             pop_order=['FF', 'E'], color_dict={'FF': 'grey', 'E': 'r'})
            plot_inferred_spike_rates(full_spike_times_dict, binned_firing_rates_dict, input_t=binned_t_edges,
                                      active_rate_threshold=active_rate_threshold)
            if subset_full_voltage_rec_dict is not None and full_rec_t is not None:
                rec_t = np.arange(0., duration, dt)
                plot_voltage_traces(subset_full_voltage_rec_dict, full_rec_t, valid_t=rec_t)

            if connectivity_type == 'gaussian':
                plot_2D_connection_distance(pop_syn_proportions, pop_cell_positions, connectivity_dict)
        print('pid: %i; Processed trial: %s, network_instance: %i, network_id: %i from file: %s in %.1f s' %
              (os.getpid(), trial_key, network_instance, network_id, data_file_path, time.time() - current_time))

    trial_averaged_firing_rate_matrix_dict = get_trial_averaged_firing_rate_matrix_dict(firing_rate_matrix_dict_list)
    centered_firing_rate_mean_dict, centered_firing_rate_sem_dict, sorted_gid_dict = \
        analyze_selectivity_from_firing_rate_matrix_dict(trial_averaged_firing_rate_matrix_dict, gid_order_dict)
    sorted_trial_averaged_firing_rate_matrix_dict = \
        sort_firing_rate_matrix_dict(trial_averaged_firing_rate_matrix_dict, gid_order_dict, sorted_gid_dict)

    sorted_connection_weights_matrix_dict = \
        sort_connection_weights_matrix_dict(connection_weights_matrix_dict, gid_order_dict, sorted_gid_dict)

    modulation_depth_dict, delta_peak_locs_dict = \
        analyze_selectivity_input_output(binned_t_edges, trial_averaged_firing_rate_matrix_dict, pop_gid_ranges,
                                         ['E', 'FF'], ['E', 'I'], connectivity_dict, connection_weights_matrix_dict)

    mean_rate_active_cells_mean_dict, mean_rate_active_cells_sem_dict, pop_fraction_active_mean_dict, \
    pop_fraction_active_sem_dict = \
        get_trial_averaged_pop_activity_stats(mean_rate_active_cells_dict_list, pop_fraction_active_dict_list)

    fft_f = next(iter(fft_f_dict.values()))
    fft_power_mean_dict, fft_power_sem_dict = get_trial_averaged_fft_power(fft_f, fft_power_dict_list)

    if 'Gamma' in filter_envelope_dict:
        fft_f_nested_gamma = next(iter(fft_f_nested_gamma_dict.values()))
        fft_power_nested_gamma_mean_dict, fft_power_nested_gamma_sem_dict = \
            get_trial_averaged_fft_power(fft_f_nested_gamma, fft_power_nested_gamma_dict_list)
    else:
        fft_f_nested_gamma = None
        fft_power_nested_gamma_mean_dict = None

    if plot:
        plot_firing_rate_heatmaps_from_matrix(sorted_trial_averaged_firing_rate_matrix_dict, binned_t_edges,
                                              gids_sorted=True, pop_order=pop_order, label_dict=label_dict,
                                              normalize_t=False)
        plot_average_selectivity(binned_t_edges, centered_firing_rate_mean_dict, centered_firing_rate_sem_dict,
                                 pop_order=pop_order, label_dict=label_dict, color_dict=color_dict)
        plot_selectivity_input_output(modulation_depth_dict, delta_peak_locs_dict, pop_order=['E', 'I'],
                                      label_dict=label_dict, color_dict=color_dict)
        plot_connection_weights_heatmaps_from_matrix(sorted_connection_weights_matrix_dict, gids_sorted=True,
                                                     pop_order=pop_order, label_dict=label_dict)
        plot_pop_activity_stats(binned_t_edges, mean_rate_active_cells_mean_dict, pop_fraction_active_mean_dict, \
                                mean_rate_active_cells_sem_dict, pop_fraction_active_sem_dict, pop_order=pop_order, \
                                label_dict=label_dict, color_dict=color_dict)
        plot_rhythmicity_psd(fft_f, fft_power_mean_dict, fft_power_sem_dict, pop_order=pop_order, label_dict=label_dict,
                             color_dict=color_dict)
        if 'Gamma' in filter_envelope_dict:
            plot_rhythmicity_psd(fft_f_nested_gamma, fft_power_nested_gamma_mean_dict,
                                 fft_power_nested_gamma_sem_dict, pop_order=pop_order, label_dict=label_dict,
                                 color_dict=color_dict)

    sys.stdout.flush()

    return binned_t_edges, sorted_trial_averaged_firing_rate_matrix_dict, sorted_gid_dict, \
           centered_firing_rate_mean_dict, mean_rate_active_cells_mean_dict, pop_fraction_active_mean_dict, fft_f, \
           fft_power_mean_dict, fft_f_nested_gamma, fft_power_nested_gamma_mean_dict, modulation_depth_dict, \
           delta_peak_locs_dict


def analyze_simple_network_replay_data_from_file(data_file_path, data_key='0', trial=0, fine_binned_dt=1.,
                                                 coarse_binned_dt=20., filter_order=None, filter_label_dict=None,
                                                 filter_color_dict=None, filter_xlim_dict=None, pop_order=None,
                                                 label_dict=None, color_dict=None, plot=True, verbose=False):
    """
    Given an hdf5 file containing multiple trials from one network instance, import data, process data, and plot example
    traces from the specified trial.
    :param data_file_path: str (path)
    :param data_key: int or str
    :param trial: int
    :param fine_binned_dt: float
    :param coarse_binned_dt: float
    :param filter_order: list of str
    :param filter_label_dict: dict
    :param filter_color_dict: dict
    :param filter_xlim_dict: dict of tuple of float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    :param plot: bool
    :param verbose: bool
    """
    if not os.path.isfile(data_file_path):
        raise IOError('analyze_simple_network_replay_data_from_file: invalid data file path: %s' % data_file_path)

    full_spike_times_dict = defaultdict(dict)
    filter_bands = dict()
    connection_weights_matrix_dict = dict()
    subset_full_voltage_rec_dict = defaultdict(dict)
    tuning_peak_locs = dict()
    pop_syn_proportions = dict()
    pop_cell_positions = dict()

    group_key = 'simple_network_exported_replay_data'
    shared_context_key = 'shared_context'
    with h5py.File(data_file_path, 'r') as f:
        group = get_h5py_group(f, [data_key, group_key])
        subgroup = group[shared_context_key]
        connectivity_type = get_h5py_attr(subgroup.attrs, 'connectivity_type')
        active_rate_threshold = subgroup.attrs['active_rate_threshold']
        duration = get_h5py_attr(subgroup.attrs, 'duration')
        buffer = get_h5py_attr(subgroup.attrs, 'buffer')
        network_id = get_h5py_attr(subgroup.attrs, 'network_id')
        network_instance = get_h5py_attr(subgroup.attrs, 'network_instance')

        if 'full_rec_t' in subgroup:
            full_rec_t = subgroup['full_rec_t'][:]
            dt = full_rec_t[1] - full_rec_t[0]
        else:
            full_rec_t = None
            dt = None

        pop_gid_ranges = dict()
        for pop_name in subgroup['pop_gid_ranges']:
            pop_gid_ranges[pop_name] = tuple(subgroup['pop_gid_ranges'][pop_name][:])
        data_group = subgroup['filter_bands']
        for this_filter in data_group:
            filter_bands[this_filter] = data_group[this_filter][:]
        data_group = subgroup['connection_weights']
        for target_pop_name in data_group:
            connection_weights_matrix_dict[target_pop_name] = dict()
            for source_pop_name in data_group[target_pop_name]:
                connection_weights_matrix_dict[target_pop_name][source_pop_name] = \
                    data_group[target_pop_name][source_pop_name][:]
        if 'tuning_peak_locs' in subgroup and len(subgroup['tuning_peak_locs']) > 0:
            data_group = subgroup['tuning_peak_locs']
            for pop_name in data_group:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(data_group[pop_name]['target_gids'], data_group[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        data_group = subgroup['pop_syn_proportions']
        for target_pop_name in data_group:
            pop_syn_proportions[target_pop_name] = dict()
            for syn_type in data_group[target_pop_name]:
                pop_syn_proportions[target_pop_name][syn_type] = dict()
                source_pop_names = data_group[target_pop_name][syn_type]['source_pop_names'][:].astype('str')
                for source_pop_name, syn_proportion in zip(source_pop_names,
                                                           data_group[target_pop_name][syn_type]['syn_proportions'][:]):
                    pop_syn_proportions[target_pop_name][syn_type][source_pop_name] = syn_proportion
        data_group = subgroup['pop_cell_positions']
        for pop_name in data_group:
            pop_cell_positions[pop_name] = dict()
            for gid, position in zip(data_group[pop_name]['gids'][:], data_group[pop_name]['positions'][:]):
                pop_cell_positions[pop_name][gid] = position

        trial_key = str(trial)
        if trial_key not in group:
            raise RuntimeError('analyze_simple_network_replay_data_from_file: data for trial: %i not found in data '
                               'file path: %s' % (trial, data_file_path))
        subgroup = get_h5py_group(group, [trial_key])
        data_group = subgroup['full_spike_times']
        for pop_name in data_group:
            for gid_key in data_group[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]
        if 'subset_full_voltage_recs' in subgroup:
            data_group = subgroup['subset_full_voltage_recs']
            for pop_name in data_group:
                for gid_key in data_group[pop_name]:
                    subset_full_voltage_rec_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]
        else:
            subset_full_voltage_rec_dict = None

    fine_buffered_binned_t_edges = \
        np.arange(-buffer, duration + buffer + fine_binned_dt / 2., fine_binned_dt)
    fine_buffered_binned_t = fine_buffered_binned_t_edges[:-1] + fine_binned_dt / 2.
    fine_binned_t_edges = np.arange(0., duration + fine_binned_dt / 2., fine_binned_dt)
    fine_binned_t = fine_binned_t_edges[:-1] + fine_binned_dt / 2.

    buffered_binned_t_edges = \
        np.arange(-buffer, duration + buffer + coarse_binned_dt / 2., coarse_binned_dt)
    buffered_binned_t = buffered_binned_t_edges[:-1] + coarse_binned_dt / 2.
    binned_t_edges = np.arange(0., duration + coarse_binned_dt / 2., coarse_binned_dt)
    binned_t = binned_t_edges[:-1] + coarse_binned_dt / 2.

    fine_buffered_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict,
                                                                        fine_buffered_binned_t_edges)
    fine_buffered_pop_mean_rate_from_binned_spike_count_dict = \
        get_pop_mean_rate_from_binned_spike_count(fine_buffered_binned_spike_count_dict, dt=fine_binned_dt)

    buffered_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, buffered_binned_t_edges)
    buffered_firing_rates_from_binned_spike_count_dict = \
        get_firing_rates_from_binned_spike_count_dict(buffered_binned_spike_count_dict, bin_dur=coarse_binned_dt,
                                                      smooth=None, wrap=False)

    buffered_firing_rate_matrix_dict, gid_order_dict = \
        get_firing_rate_matrix_dict_from_nested_dict(buffered_firing_rates_from_binned_spike_count_dict, buffered_binned_t)
    sorted_gid_dict = get_sorted_gid_dict_from_tuning_peak_locs(gid_order_dict, tuning_peak_locs)
    sorted_buffered_firing_rate_matrix_dict = sort_firing_rate_matrix_dict(buffered_firing_rate_matrix_dict,
                                                                           gid_order_dict, sorted_gid_dict)

    mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict = \
        get_pop_activity_stats(buffered_firing_rates_from_binned_spike_count_dict, input_t=buffered_binned_t,
                               threshold=active_rate_threshold)

    fft_f_dict, fft_power_dict, filter_psd_f_dict, filter_psd_power_dict, filter_envelope_dict, \
    filter_envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict = \
        get_pop_bandpass_filtered_signal_stats(fine_buffered_pop_mean_rate_from_binned_spike_count_dict,
                                               filter_bands, input_t=fine_buffered_binned_t,
                                               valid_t=fine_buffered_binned_t, output_t=fine_binned_t, pad=False,
                                               filter_order=filter_order, filter_label_dict=filter_label_dict,
                                               filter_color_dict=filter_color_dict,
                                               filter_xlim_dict=filter_xlim_dict, pop_order=pop_order,
                                               label_dict=label_dict, color_dict=color_dict, plot=plot)
    fft_f = next(iter(fft_f_dict.values()))
    sorted_connection_weights_matrix_dict = \
        sort_connection_weights_matrix_dict(connection_weights_matrix_dict, gid_order_dict, sorted_gid_dict)

    if plot:
        if subset_full_voltage_rec_dict is not None and full_rec_t is not None:
            rec_t = np.arange(0., duration, dt)
            plot_voltage_traces(subset_full_voltage_rec_dict, full_rec_t, valid_t=rec_t)
        plot_firing_rate_heatmaps_from_matrix(sorted_buffered_firing_rate_matrix_dict, buffered_binned_t,
                                              gids_sorted=True, pop_order=pop_order, label_dict=label_dict,
                                              normalize_t=False)
        plot_connection_weights_heatmaps_from_matrix(sorted_connection_weights_matrix_dict, gids_sorted=True,
                                                     pop_order=pop_order, label_dict=label_dict)
        plot_pop_activity_stats(buffered_binned_t_edges, mean_rate_active_cells_dict, pop_fraction_active_dict, \
                                pop_order=pop_order, \
                                label_dict=label_dict, color_dict=color_dict)

        plot_rhythmicity_psd(fft_f, fft_power_dict, pop_order=pop_order, label_dict=label_dict,
                             color_dict=color_dict)
        plot_replay_spike_rasters(full_spike_times_dict, binned_t_edges, sorted_gid_dict, trial_key=trial_key, \
                                      pop_order=pop_order, label_dict=label_dict)

    return fft_f, fft_power_dict


def circular_linear_fit_error(p, x, y, end=1.):
    p_y = p[0] * x + p[1]
    err = 0.
    for i in range(len(y)):
        err += 2. * (1. - np.cos(2. * np.pi / end * (y[i] - p_y[i])))
    return err


def get_circular_linear_pos(p, x, end=1.):
    return np.mod(p[0] * x + p[1], end)


def fit_trajectory_slope(bins, this_trial_pos, plot=False):
    from scipy.optimize import basinhopping
    from scipy.stats import pearsonr
    bounds = ((-5., 5.), (-1.5, 1.5))
    stepsize = 5.
    result = basinhopping(circular_linear_fit_error, [0., 0.5],
                          minimizer_kwargs={'args': (bins, this_trial_pos), 'method': 'L-BFGS-B', 'bounds': bounds},
                          stepsize=stepsize)
    fit_pos = get_circular_linear_pos(result.x, bins)
    r, p = pearsonr(this_trial_pos, fit_pos)
    if plot:
        fig = plt.figure()
        plt.plot(bins, this_trial_pos)
        plt.plot(bins, fit_pos)
        plt.title('Slope: %.2E, p: %.4f' % (result.x[0], p))
        fig.show()
    return result.x[0], p


def decode_position(binned_spike_count_matrix_dict, template_firing_rate_matrix_dict, bin_dur=20.):
    """

    :param binned_spike_count_matrix_dict: dict of 2d array
    :param template_firing_rate_matrix_dict: dict of 2d array
    :param bin_dur: float
    :return: dict of 2d array
    """
    import numpy.matlib
    p_pos_dict = dict()
    for pop_name in binned_spike_count_matrix_dict:
        if binned_spike_count_matrix_dict[pop_name].shape[0] != template_firing_rate_matrix_dict[pop_name].shape[0]:
            raise RuntimeError('decode_position_from_offline_replay: population: %s; mismatched number of cells to'
                               ' decode')
        binned_spike_count = binned_spike_count_matrix_dict[pop_name]
        template_firing_rates = template_firing_rate_matrix_dict[pop_name] + 0.1  # small offset to avoid veto by zero rate

        p_pos = np.empty((template_firing_rates.shape[1], binned_spike_count.shape[1]))
        p_pos.fill(np.nan)

        population_spike_count = np.exp(-bin_dur / 1000. * np.sum(template_firing_rates, axis=0, dtype='float128'))
        for index in range(binned_spike_count.shape[1]):
            local_spike_count_array = binned_spike_count[:, index].astype('float128')
            if np.sum(local_spike_count_array) > 0.:
                n = np.matlib.repmat(local_spike_count_array, template_firing_rates.shape[1], 1).T
                this_p_pos = (template_firing_rates ** n).prod(axis=0) * population_spike_count
                this_p_sum = np.nansum(this_p_pos)
                if np.isnan(this_p_sum):
                    p_pos[:, index] = np.nan
                elif this_p_sum > 0.:
                    p_pos[:, index] = this_p_pos / this_p_sum
                else:
                    p_pos[:, index] = np.nan
        p_pos_dict[pop_name] = p_pos

    return p_pos_dict


def baks(spktimes, time, a=1.5, b=None):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)
    BAKS is a method for estimating firing rate from spike train data that uses kernel smoothing technique
    with adaptive bandwidth determined using a Bayesian approach
    ---------------INPUT---------------
    - spktimes : spike event times [s]
    - time : time points at which the firing rate is estimated [s]
    - a : shape parameter (alpha)
    - b : scale parameter (beta)
    ---------------OUTPUT---------------
    - rate : estimated firing rate [nTime x 1] (Hz)
    - h : adaptive bandwidth [nTime x 1]

    Based on "Estimation of neuronal firing rate using Bayesian adaptive kernel smoother (BAKS)"
    https://github.com/nurahmadi/BAKS
    """
    from scipy.special import gamma
    rate = np.zeros((len(time),))

    n = len(spktimes)
    if n < 1:
        return rate, None

    sumnum = 0
    sumdenom = 0

    if b is None:
        b = 0.42
    b = float(n) ** b

    for i in range(n):
        numerator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a)
        denominator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a - 0.5)
        sumnum = sumnum + numerator
        sumdenom = sumdenom + denominator

    h = (gamma(a) / gamma(a + 0.5)) * (sumnum / sumdenom)

    for j in range(n):
        K = (1. / (np.sqrt(2. * np.pi) * h)) * np.exp(-((time - spktimes[j]) ** 2) / (2. * h ** 2))
        rate = rate + K

    return rate, h


def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'np.random.RandomState()'
    :return: list of m spike times (ms)
    """
    if generator is None:
        generator = random
    interp_t = np.arange(t[0], t[-1] + dt, dt)
    try:
        interp_rate = np.interp(interp_t, t, rate)
    except Exception as e:
        print('t shape: %s rate shape: %s' % (str(t.shape), str(rate.shape)))
        sys.stdout.flush()
        time.sleep(0.1)
        raise(e)
    interp_rate /= 1000.
    spike_times = []
    non_zero = np.where(interp_rate > 1.e-100)[0]
    if len(non_zero) == 0:
        return spike_times
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    max_rate = np.max(interp_rate)
    if not max_rate > 0.:
        return spike_times
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.uniform(0.0, 1.0)
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI / dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.uniform(0.0, 1.0) <= (interp_rate[i] / max_rate)) and \
                    ISI_memory >= 0.:
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return spike_times


def merge_connection_weights_dicts(target_gid_dict_list, weights_dict_list):
    """

    :param target_gid_dict_list: list of dict: {'pop_name': array of int}
    :param weights_dict_list: list of nested dict: {'target_pop_name': {'source_pop_name': 2D array of float} }
    :return: nested dict
    """
    connection_weights_dict = dict()
    unsorted_target_gid_dict = defaultdict(list)

    for k, target_gid_dict in enumerate(target_gid_dict_list):
        weights_dict = weights_dict_list[k]
        for target_pop_name in target_gid_dict:
            unsorted_target_gid_dict[target_pop_name].extend(target_gid_dict[target_pop_name])
            if target_pop_name not in connection_weights_dict:
                connection_weights_dict[target_pop_name] = dict()
            for source_pop_name in weights_dict[target_pop_name]:
                if source_pop_name not in connection_weights_dict[target_pop_name]:
                    connection_weights_dict[target_pop_name][source_pop_name] = \
                        [weights_dict[target_pop_name][source_pop_name]]
                else:
                    connection_weights_dict[target_pop_name][source_pop_name].append(
                        weights_dict[target_pop_name][source_pop_name])
    for target_pop_name in unsorted_target_gid_dict:
        sorted_target_gid_indexes = np.argsort(unsorted_target_gid_dict[target_pop_name])
        for source_pop_name in connection_weights_dict[target_pop_name]:
            connection_weights_dict[target_pop_name][source_pop_name] = \
                np.concatenate(connection_weights_dict[target_pop_name][source_pop_name])
            connection_weights_dict[target_pop_name][source_pop_name][:] = \
                connection_weights_dict[target_pop_name][source_pop_name][sorted_target_gid_indexes, :]

    return connection_weights_dict


def analyze_decoded_trajectory_run_data(decoded_pos_matrix_dict, actual_position, decode_duration):
    """

    :param decoded_pos_matrix_dict: dict of array of float (num_trials, num_position_bins)
    :param actual_position: array of float (num_bins)
    :param decode_duration: float
    :return: tuple of dict of array of float
    """
    decoded_pos_error_mean_dict = {}
    decoded_pos_error_sem_dict = {}
    sequence_len_mean_dict = {}
    sequence_len_sem_dict = {}

    for pop_name in decoded_pos_matrix_dict:
        if pop_name not in decoded_pos_error_mean_dict:
            decoded_pos_error_mean_dict[pop_name] = []
            sequence_len_mean_dict[pop_name] = []
        this_decoded_pos_matrix = decoded_pos_matrix_dict[pop_name][:, :] / decode_duration
        num_trials = this_decoded_pos_matrix.shape[0]
        for trial in range(num_trials):
            this_trial_pos = this_decoded_pos_matrix[trial, :]
            this_trial_error = np.subtract(actual_position / decode_duration, this_trial_pos)
            if np.all(np.isnan(this_trial_error)):
                continue
            for i in range(len(this_trial_error)):
                if np.isnan(this_trial_error[i]):
                    if i == 0:
                        j = np.where(~np.isnan(this_trial_error))[0][0]
                    else:
                        j = np.where(~np.isnan(this_trial_error[:i]))[0][-1]
                    this_trial_error[i] = this_trial_error[j]
            this_trial_error[np.where(this_trial_error < -0.5)] += 1.
            this_trial_error[np.where(this_trial_error > 0.5)] -= 1.
            analytic_signal = hilbert(this_trial_error)
            amplitude_envelope = np.abs(analytic_signal)

            decoded_pos_error_mean_dict[pop_name].append(np.abs(this_trial_error))
            sequence_len_mean_dict[pop_name].append(2. * amplitude_envelope)
        decoded_pos_error_sem_dict[pop_name] = np.std(decoded_pos_error_mean_dict[pop_name], axis=0) / \
                                               np.sqrt(num_trials)
        decoded_pos_error_mean_dict[pop_name] = np.mean(decoded_pos_error_mean_dict[pop_name], axis=0)
        sequence_len_sem_dict[pop_name] = np.std(sequence_len_mean_dict[pop_name], axis=0) / np.sqrt(num_trials)
        sequence_len_mean_dict[pop_name] = np.mean(sequence_len_mean_dict[pop_name], axis=0)

    return decoded_pos_error_mean_dict, decoded_pos_error_sem_dict, sequence_len_mean_dict, sequence_len_sem_dict


def plot_decoded_trajectory_run_data(decoded_pos_error_mean_dict, sequence_len_mean_dict, actual_position,
                                     decode_duration, decoded_pos_error_sem_dict=None, sequence_len_sem_dict=None,
                                     decoded_pos_error_ymax=None, sequence_len_ymax=None, pop_order=None,
                                     label_dict=None, color_dict=None):
    """

    :param decoded_pos_error_mean_dict: dict of array of float (num_position_bins)
    :param sequence_len_mean_dict: dict of array of float (num_position_bins)
    :param actual_position: array of float (num_position_bins)
    :param decode_duration: float
    :param decoded_pos_error_sem_dict: dict of array of float (num_position_bins)
    :param sequence_len_sem_dict: dict of array of float (num_position_bins)
    :param decoded_pos_error_ymax: float
    :param sequence_len_ymax: float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    """
    normalized_position = actual_position / decode_duration
    if pop_order is None:
        pop_order = sorted(list(decoded_pos_error_mean_dict.keys()))

    fig, axes = plt.subplots(2, figsize=(4., 7.))

    ymin0 = 0.
    ymin1 = 0.
    for pop_name in pop_order:
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        if color_dict is not None:
            color = color_dict[pop_name]
            axes[0].plot(normalized_position, decoded_pos_error_mean_dict[pop_name], label=label, color=color)
            if decoded_pos_error_sem_dict is not None:
                axes[0].fill_between(normalized_position,
                                     decoded_pos_error_mean_dict[pop_name] - decoded_pos_error_sem_dict[pop_name],
                                     decoded_pos_error_mean_dict[pop_name] + decoded_pos_error_sem_dict[pop_name],
                                     alpha=0.25, linewidth=0, color=color)
                ymin0 = min(0., np.min(decoded_pos_error_mean_dict[pop_name] - decoded_pos_error_sem_dict[pop_name]))
            else:
                ymin0 = min(0., np.min(decoded_pos_error_mean_dict[pop_name]))
            axes[1].plot(normalized_position, sequence_len_mean_dict[pop_name], label=label, color=color)
            if sequence_len_sem_dict is not None:
                axes[1].fill_between(normalized_position,
                                     sequence_len_mean_dict[pop_name] - sequence_len_sem_dict[pop_name],
                                     sequence_len_mean_dict[pop_name] + sequence_len_sem_dict[pop_name],
                                     alpha=0.25, linewidth=0, color=color)
                ymin1 = min(0., np.min(sequence_len_mean_dict[pop_name] - sequence_len_sem_dict[pop_name]))
            else:
                ymin1 = min(0., np.min(sequence_len_mean_dict[pop_name]))
        else:
            axes[0].plot(normalized_position, decoded_pos_error_mean_dict[pop_name], label=label)
            if decoded_pos_error_sem_dict is not None:
                axes[0].fill_between(normalized_position,
                                     decoded_pos_error_mean_dict[pop_name] - decoded_pos_error_sem_dict[pop_name],
                                     decoded_pos_error_mean_dict[pop_name] + decoded_pos_error_sem_dict[pop_name],
                                     alpha=0.25, linewidth=0)
                ymin0 = min(0., np.min(decoded_pos_error_mean_dict[pop_name] - decoded_pos_error_sem_dict[pop_name]))
            else:
                ymin0 = min(0., np.min(decoded_pos_error_mean_dict[pop_name]))
            axes[1].plot(normalized_position, sequence_len_mean_dict[pop_name], label=label)
            if sequence_len_sem_dict is not None:
                axes[1].fill_between(normalized_position,
                                     sequence_len_mean_dict[pop_name] - sequence_len_sem_dict[pop_name],
                                     sequence_len_mean_dict[pop_name] + sequence_len_sem_dict[pop_name],
                                     alpha=0.25, linewidth=0)
                ymin1 = min(0., np.min(sequence_len_mean_dict[pop_name] - sequence_len_sem_dict[pop_name]))
            else:
                ymin1 = min(0., np.min(sequence_len_mean_dict[pop_name]))
    axes[0].set_title('Decoded position error', fontsize=mpl.rcParams['font.size'])
    axes[0].set_ylabel('Fraction of track length')
    axes[0].set_xlabel('Normalized position')
    axes[0].legend(loc='best', frameon=False, fontsize=mpl.rcParams['font.size'], handlelength=1)
    axes[0].set_xlim((0., 1.))
    if decoded_pos_error_ymax is None:
        axes[0].set_ylim((ymin0, axes[0].get_ylim()[1]))
    else:
        axes[0].set_ylim((ymin0, decoded_pos_error_ymax))
    axes[1].set_title('Theta sequence length', fontsize=mpl.rcParams['font.size'])
    axes[1].set_ylabel('Fraction of track length')
    axes[1].set_xlabel('Normalized position')
    if sequence_len_ymax is None:
        axes[1].set_ylim((ymin1, axes[1].get_ylim()[1]))
    else:
        axes[1].set_ylim((ymin1, sequence_len_ymax))
    axes[1].set_xlim((0., 1.))
    clean_axes(axes)
    fig.tight_layout()
    fig.show()


def analyze_decoded_trajectory_run_data_across_instances(decoded_pos_error_mean_dict_list, sequence_len_mean_dict_list):
    """
    Given lists of dicts containing data from different network instances, return dicts containing the mean and sem
    for decoded position error and sequence length.
    :param decoded_pos_error_mean_dict_list: list of dict of array of float
    :param sequence_len_mean_dict_list: list of dict of array of float
    :return: tuple of dict of array of float
    """
    decoded_pos_error_mean_dict = {}
    decoded_pos_error_sem_dict = {}
    sequence_len_mean_dict = {}
    sequence_len_sem_dict = {}

    num_instances = len(decoded_pos_error_mean_dict_list)
    for i, decoded_pos_error_mean_dict_instance in enumerate(decoded_pos_error_mean_dict_list):
        sequence_len_mean_dict_instance = sequence_len_mean_dict_list[i]
        for pop_name in decoded_pos_error_mean_dict_instance:
            if pop_name not in decoded_pos_error_mean_dict:
                decoded_pos_error_mean_dict[pop_name] = []
                sequence_len_mean_dict[pop_name] = []
            decoded_pos_error_mean_dict[pop_name].append(decoded_pos_error_mean_dict_instance[pop_name])
            sequence_len_mean_dict[pop_name].append(sequence_len_mean_dict_instance[pop_name])
    for pop_name in decoded_pos_error_mean_dict:
        decoded_pos_error_sem_dict[pop_name] = np.std(decoded_pos_error_mean_dict[pop_name], axis=0) / \
                                               np.sqrt(num_instances)
        decoded_pos_error_mean_dict[pop_name] = np.mean(decoded_pos_error_mean_dict[pop_name], axis=0)
        sequence_len_sem_dict[pop_name] = np.std(sequence_len_mean_dict[pop_name], axis=0) / \
                                          np.sqrt(num_instances)
        sequence_len_mean_dict[pop_name] = np.mean(sequence_len_mean_dict[pop_name], axis=0)

    return decoded_pos_error_mean_dict, decoded_pos_error_sem_dict, sequence_len_mean_dict, sequence_len_sem_dict


def load_decoded_data(decode_data_file_path, export_data_key):
    """

    :param decode_data_file_path: str (path)
    :param export_data_key: str
    :return: dict: {pop_name: 2D array}
    """
    group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    decoded_pos_matrix_dict = dict()
    with h5py.File(decode_data_file_path, 'a') as f:
        group = get_h5py_group(f, [export_data_key, group_key, shared_context_key])
        subgroup = get_h5py_group(group, ['decoded_pos_matrix'])
        for pop_name in subgroup:
            decoded_pos_matrix_dict[pop_name] = subgroup[pop_name][:,:]

    return decoded_pos_matrix_dict


def analyze_decoded_trajectory_replay_data(decoded_pos_matrix_dict_list, bin_dur, template_duration,
                                           path_len_criterion=1., max_step_criterion=0.35,
                                           fraction_run_velocity_criterion=0.5):
    """

    :param decoded_pos_matrix_dict_list: list of dict: {pop_name: array (num_trials, num_bins)
    :param bin_dur: float (ms)
    :param template_duration: float (ms)
    :param path_len_criterion: float (fraction of template)
    :param max_step_criterion: flaat (fraction of template)
    :param fraction_run_velocity_criterion: float (fraction of template velocity)
    :return: tuple of dicts
    """
    all_decoded_pos_instances_list_dict = defaultdict(list)
    decoded_path_len_instances_list_dict = defaultdict(list)
    decoded_velocity_instances_list_dict = defaultdict(list)
    decoded_max_step_instances_list_dict = defaultdict(list)
    met_criterion_fraction_instances_list_dict = defaultdict(list)

    for decoded_pos_matrix_dict in decoded_pos_matrix_dict_list:
        for pop_name in decoded_pos_matrix_dict:
            this_decoded_pos_matrix = decoded_pos_matrix_dict[pop_name][:, :] / template_duration
            trial_dur = this_decoded_pos_matrix.shape[1] * bin_dur / 1000.
            clean_indexes = ~np.isnan(this_decoded_pos_matrix)
            all_decoded_pos_array = this_decoded_pos_matrix[clean_indexes]
            decoded_path_len_list = []
            decoded_velocity_list = []
            decoded_max_step_list = []
            excluded_count = 0
            for trial in range(this_decoded_pos_matrix.shape[0]):
                excluded = False
                this_trial_pos = this_decoded_pos_matrix[trial, :]
                clean_indexes = ~np.isnan(this_trial_pos)
                if len(clean_indexes) > 0:
                    this_trial_diff = np.diff(this_trial_pos[clean_indexes])
                    this_trial_diff[np.where(this_trial_diff < -0.5)] += 1.
                    this_trial_diff[np.where(this_trial_diff > 0.5)] -= 1.
                    this_path_len = np.sum(np.abs(this_trial_diff))
                    decoded_path_len_list.append(this_path_len)
                    if this_path_len > path_len_criterion:
                        excluded = True
                    if len(clean_indexes) < len(this_trial_pos):
                        excluded = True
                    this_trial_velocity = np.sum(this_trial_diff) / trial_dur
                    decoded_velocity_list.append(this_trial_velocity)
                    if np.abs(this_trial_velocity) < 1. / (template_duration / 1000.) * fraction_run_velocity_criterion:
                        excluded = True
                    if len(this_trial_diff) > 0:
                        this_max_step = np.max(np.abs(this_trial_diff))
                        if this_max_step > max_step_criterion:
                            excluded = True
                        decoded_max_step_list.append(this_max_step)
                else:
                    excluded = True
                if excluded:
                    excluded_count += 1
            all_decoded_pos_instances_list_dict[pop_name].append(all_decoded_pos_array)
            decoded_path_len_instances_list_dict[pop_name].append(decoded_path_len_list)
            decoded_velocity_instances_list_dict[pop_name].append(decoded_velocity_list)
            decoded_max_step_instances_list_dict[pop_name].append(decoded_max_step_list)
            met_criterion_fraction_instances_list_dict[pop_name].append(
                1. - excluded_count / this_decoded_pos_matrix.shape[0])

    return all_decoded_pos_instances_list_dict, decoded_path_len_instances_list_dict, \
           decoded_velocity_instances_list_dict, decoded_max_step_instances_list_dict, \
           met_criterion_fraction_instances_list_dict


def plot_decoded_trajectory_replay_data(decoded_pos_matrix_dict, bin_dur, template_duration, pop_order=None,
                                        label_dict=None, color_dict=None):
    """

    :param decoded_pos_matrix_dict: dict or list of dict: {pop_name: 2d array of float}
    :param bin_dur: float
    :param template_duration: float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param color_dict: dict; {pop_name: str}
    """
    if not isinstance(decoded_pos_matrix_dict, list):
        decoded_pos_matrix_dict_instances_list = [decoded_pos_matrix_dict]
    else:
        decoded_pos_matrix_dict_instances_list = decoded_pos_matrix_dict

    all_decoded_pos_instances_list_dict, decoded_path_len_instances_list_dict, \
    decoded_velocity_instances_list_dict, decoded_max_step_instances_list_dict, \
    met_criterion_fraction_instances_list_dict = \
        analyze_decoded_trajectory_replay_data(decoded_pos_matrix_dict_instances_list, bin_dur, template_duration)

    if pop_order is None:
        pop_order = sorted(list(all_decoded_pos_instances_list_dict.keys()))

    fig, axes = plt.subplots(1, 5, figsize=(2.77 * 5, 3.2))  # , constrained_layout=True)
    flat_axes = axes.flatten()

    max_path_len = np.nanmax(list(decoded_path_len_instances_list_dict.values()))
    max_vel_mean = np.nanmax(list(decoded_velocity_instances_list_dict.values()))
    min_vel_mean = np.nanmin(list(decoded_velocity_instances_list_dict.values()))
    max_step_val = np.nanmax([np.nanmax(instance) for pop_name in decoded_max_step_instances_list_dict for instance in
                              decoded_max_step_instances_list_dict[pop_name]])
    max_sequence_fraction = np.max(list(met_criterion_fraction_instances_list_dict.values()))

    num_instances = len(decoded_pos_matrix_dict_instances_list)
    for pop_name in pop_order:
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        if color_dict is not None:
            color = color_dict[pop_name]
        else:
            color = None

        axis = 0
        hist_list = []
        for all_decoded_pos in all_decoded_pos_instances_list_dict[pop_name]:
            hist, edges = np.histogram(all_decoded_pos, bins=np.linspace(0., 1., 21), density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            flat_axes[axis].plot(edges[1:] - bin_width / 2., hist_list[0], label=label, color=color)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            flat_axes[axis].plot(edges[1:] - bin_width / 2., mean_hist, label=label, color=color)
            flat_axes[axis].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0, color=color)

        axis += 1
        hist_list = []
        for decoded_path_len in decoded_path_len_instances_list_dict[pop_name]:
            hist, edges = np.histogram(decoded_path_len, bins=np.linspace(0., max_path_len, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            flat_axes[axis].plot(edges[1:] - bin_width / 2., hist_list[0], label=label, color=color)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            flat_axes[axis].plot(edges[1:] - bin_width / 2., mean_hist, label=label, color=color)
            flat_axes[axis].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0, color=color)
        axis += 1
        hist_list = []
        for decoded_velocity in decoded_velocity_instances_list_dict[pop_name]:
            hist, edges = np.histogram(decoded_velocity, bins=np.linspace(min_vel_mean, max_vel_mean, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            flat_axes[axis].plot(edges[1:] - bin_width / 2., hist_list[0], label=label, color=color)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            flat_axes[axis].plot(edges[1:] - bin_width / 2., mean_hist, label=label, color=color)
            flat_axes[axis].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0, color=color)
        """
        axis += 1
        hist_list = []
        for decoded_max_step in decoded_max_step_instances_list_dict[pop_name]:
            hist, edges = np.histogram(decoded_max_step, bins=np.linspace(0., max_step_val, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            flat_axes[axis].plot(edges[1:] - bin_width / 2., hist_list[0], label=label, color=color)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            flat_axes[axis].plot(edges[1:] - bin_width / 2., mean_hist, label=label, color=color)
            flat_axes[axis].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                         alpha=0.25, linewidth=0, color=color)
        """

    axis += 1
    pos_start = 0
    xlabels = []
    for pop_name in pop_order:
        if label_dict is not None:
            label = label_dict[pop_name]
        else:
            label = pop_name
        xlabels.append(label)

        bp = flat_axes[axis].boxplot(met_criterion_fraction_instances_list_dict[pop_name],
                                     positions=[pos_start], patch_artist=True, showfliers=False)
        if color_dict[pop_name] is not None:
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=color_dict[pop_name])
        for patch in bp['boxes']:
            patch.set(facecolor='white')
        pos_start += 1
    flat_axes[axis].set_xticks(range(len(pop_order)))
    flat_axes[axis].set_xticklabels(xlabels)
    flat_axes[axis].set_ylabel('Fraction of events')
    flat_axes[axis].set_ylim(0., max(flat_axes[axis].get_ylim()[1] * 1.1, 0.5))
    flat_axes[axis].set_title('Continuous sequences', y=1.05, fontsize=mpl.rcParams['font.size'])

    axis = 0
    flat_axes[axis].set_xlim((0., 1.))
    flat_axes[axis].set_ylim((0., max(flat_axes[axis].get_ylim()[1], 0.2)))
    flat_axes[axis].set_xlabel('Normalized position')
    flat_axes[axis].set_ylabel('Fraction of events')
    flat_axes[axis].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    flat_axes[axis].set_title('Decoded positions', y=1.05, fontsize=mpl.rcParams['font.size'])

    axis += 1
    flat_axes[axis].set_xlim((0., max_path_len))
    flat_axes[axis].set_ylim((0., flat_axes[axis].get_ylim()[1]))
    flat_axes[axis].set_xlabel('Fraction of track')
    flat_axes[axis].set_ylabel('Fraction of events')
    flat_axes[axis].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    flat_axes[axis].set_title('Sequence length', y=1.05, fontsize=mpl.rcParams['font.size'])

    """
    axis += 1
    flat_axes[axis].set_xlim((0., max_step_val))
    flat_axes[axis].set_ylim((0., flat_axes[axis].get_ylim()[1]))
    flat_axes[axis].set_xlabel('Fraction of track')
    flat_axes[axis].set_ylabel('Fraction of events')
    flat_axes[axis].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    flat_axes[axis].set_title('Maximum step size', y=1.05, fontsize=mpl.rcParams['font.size'])
    """

    axis += 1
    flat_axes[axis].set_xlim((min_vel_mean, max_vel_mean))
    flat_axes[axis].set_ylim((0., flat_axes[axis].get_ylim()[1]))
    flat_axes[axis].set_xlabel('Velocity\n(Fraction of track/s)')
    flat_axes[axis].set_ylabel('Fraction of events')
    flat_axes[axis].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    flat_axes[axis].set_title('Sequence velocity', y=1.05, fontsize=mpl.rcParams['font.size'])

    clean_axes(flat_axes)
    fig.tight_layout(w_pad=0.25)
    # fig.set_constrained_layout_pads(hspace=0.15, wspace=0.1)
    fig.show()


def get_sorted_gid_dict_from_tuning_peak_locs(gid_order_dict, tuning_peak_locs):
    """

    :param gid_order_dict: dict of array of int
    :param tuning_peak_locs: nested dict: {pop_name (str): {gid (int): float}}
    :return: dict of array of int
    """
    sorted_gid_dict = dict()
    for pop_name in gid_order_dict:
        if pop_name in tuning_peak_locs:
            this_target_gids = np.array(list(tuning_peak_locs[pop_name].keys()))
            this_peak_locs = np.array(list(tuning_peak_locs[pop_name].values()))
            indexes = np.argsort(this_peak_locs)
            sorted_gid_dict[pop_name] = this_target_gids[indexes]
        else:
            sorted_gid_dict[pop_name] = np.copy(gid_order_dict[pop_name])

    return sorted_gid_dict


def scattered_boxplot(ax, x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None,
                      showfliers="unif",
                      hide_points_within_whiskers=False,
                      boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, *, data=None):
    if showfliers=="classic":
        classic_fliers=True
    else:
        classic_fliers=False
    ax.boxplot(x, notch=notch, sym=sym, vert=vert, whis=whis, positions=positions, widths=widths, patch_artist=patch_artist, bootstrap=bootstrap, usermedians=usermedians, conf_intervals=conf_intervals, meanline=meanline, showmeans=showmeans, showcaps=showcaps, showbox=showbox,
               showfliers=classic_fliers,
               boxprops=boxprops, labels=labels, flierprops=flierprops, medianprops=medianprops, meanprops=meanprops, capprops=capprops, whiskerprops=whiskerprops, manage_ticks=manage_ticks, autorange=autorange, zorder=zorder,data=data)
    N=len(x)
    datashape_message = ("List of boxplot statistics and `{0}` "
                             "values must have same the length")
    # check position
    if positions is None:
        positions = list(range(1, N + 1))
    elif len(positions) != N:
        raise ValueError(datashape_message.format("positions"))

    positions = np.array(positions)
    if len(positions) > 0 and not isinstance(positions[0], Number):
        raise TypeError("positions should be an iterable of numbers")

    # width
    if widths is None:
        widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
    elif np.isscalar(widths):
        widths = [widths] * N
    elif len(widths) != N:
        raise ValueError(datashape_message.format("widths"))

    if hide_points_within_whiskers:
        import matplotlib.cbook as cbook
        from matplotlib import rcParams
        if whis is None:
            whis = rcParams['boxplot.whiskers']
        if bootstrap is None:
            bootstrap = rcParams['boxplot.bootstrap']
        bxpstats = cbook.boxplot_stats(x, whis=whis, bootstrap=bootstrap,
                                       labels=labels, autorange=autorange)
    for i in range(N):
        if hide_points_within_whiskers:
            xi=bxpstats[i]['fliers']
        else:
            xi=x[i]
        if showfliers=="unif":
            jitter=np.random.uniform(-widths[i]*0.5,widths[i]*0.5,size=np.size(xi))
        elif showfliers=="normal":
            jitter=np.random.normal(loc=0.0, scale=widths[i]*0.1,size=np.size(xi))
        elif showfliers==False or showfliers=="classic":
            return
        else:
            raise NotImplementedError("showfliers='"+str(showfliers)+"' is not implemented. You can choose from 'unif', 'normal', 'classic' and False")

        ax.scatter(positions[i]+jitter,xi,alpha=0.2,marker="o", facecolors='none', edgecolors="k")