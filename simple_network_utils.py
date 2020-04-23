from nested.utils import *
from neuron import h
from scipy.signal import butter, sosfiltfilt, sosfreqz, hilbert, periodogram, savgol_filter, hann
from collections import namedtuple, defaultdict


# Based on http://modeldb.yale.edu/39948
izhi_cell_type_param_names = ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c', 'd', 'celltype']
izhi_cell_type_params = namedtuple('izhi_cell_type_params', izhi_cell_type_param_names)
izhi_cell_type_param_dict = {
    'RS': izhi_cell_type_params(C=1., k=0.7, vr=-65., vt=-50., vpeak=35., a=0.03, b=-2., c=-55., d=100.,
                                celltype=1),
    'IB': izhi_cell_type_params(C=1.5, k=1.2, vr=-75., vt=-45., vpeak=50., a=0.01, b=5., c=-56., d=130.,
                                celltype=2),
    'CH': izhi_cell_type_params(C=0.5, k=1.5, vr=-60., vt=-40., vpeak=25., a=0.03, b=1., c=-40., d=150.,
                                celltype=3),
    'LTS': izhi_cell_type_params(C=1.0, k=1.0, vr=-56., vt=-42., vpeak=40., a=0.03, b=8., c=-53., d=20.,
                                 celltype=4),
    'FS': izhi_cell_type_params(C=0.2, k=1., vr=-55., vt=-40., vpeak=25., a=0.2, b=-2., c=-45., d=-55.,
                                celltype=5),
    'TC': izhi_cell_type_params(C=2.0, k=1.6, vr=-60., vt=-50., vpeak=35., a=0.01, b=15., c=-60., d=10.,
                                celltype=6),
    'RTN': izhi_cell_type_params(C=0.4, k=0.25, vr=-65., vt=-45., vpeak=0., a=0.015, b=10., c=-55., d=50.,
                                 celltype=7)
}
izhi_cell_types = list(izhi_cell_type_param_dict.keys())


default_syn_mech_names = \
    {'E': 'SatExp2Syn',
     'I': 'SatExp2Syn'
     }

default_syn_mech_param_rules = \
    {'SatExp2Syn': {'mech_file': 'sat_exp2syn.mod',
                    'mech_params': ['sat', 'dur_onset', 'tau_offset', 'e'],
                    'netcon_params': {'weight': 0, 'g_unit': 1}
                    },
     'ExpSyn': {'mech_file': 'expsyn.mod',
                'mech_params': ['tau', 'e'],
                'netcon_params': {'weight': 0}
                }
     }

default_syn_type_mech_params = \
    {'E': {'sat': 0.9,
           'dur_onset': 1.,  # ms
           'tau_offset': 5.,  # ms
           'g_unit': 1.,  # uS
           'e': 0.,  # mV
           'weight': 1.
           },
     'I': {'sat': 0.9,
           'dur_onset': 1.,  # ms
           'tau_offset': 10.,  # ms
           'g_unit': 1.,  # uS
           'e': -80.,  # mV
           'weight': 1.
           }
     }


class SimpleNetwork(object):

    def __init__(self, pc, pop_sizes, pop_gid_ranges, pop_cell_types, pop_syn_counts, pop_syn_proportions,
                 connection_weights_mean, connection_weights_norm_sigma, syn_mech_params, syn_mech_names=None,
                 syn_mech_param_rules=None, syn_mech_param_defaults=None, tstop=2250, duration=1000., buffer=500.,
                 equilibrate=250., dt=0.025, delay=1., v_init=-65., verbose=1, debug=False):
        """

        :param pc: ParallelContext object
        :param pop_sizes: dict of int: cell population sizes
        :param pop_gid_ranges: dict of tuple of int: start and stop indexes; gid range of each cell population
        :param pop_cell_types: dict of str: cell_type of each cell population
        :param pop_syn_counts: dict of int: number of synapses onto each cell population
        :param pop_syn_proportions: nested dict of float:
                    {target_pop_name (str): {syn_type (str): {source_pop_name (str): proportion of synapses from
                        source_pop_name population } } }
        :param connection_weights_mean: nested dict of float: mean strengths of each connection type
        :param connection_weights_norm_sigma: nested dict of float: variances of connection strengths, normalized to
                                                mean
        :param syn_mech_params: nested dict: {target_pop_name (str): {source_pop_name (str): {param_name (str): float}}}
        :param syn_mech_names: dict: {syn_name (str): name of hoc point process (str)}
        :param syn_mech_param_rules: nested dict
        :param syn_mech_param_defaults: nested dict
        :param tstop: int: full simulation duration, including equilibration period and buffer (ms)
        :param duration: float: simulation duration (ms)
        :param buffer: float: duration of simulation buffer at start and end (ms)
        :param equilibrate: float: duration of simulation equilibration period at start (ms)
        :param dt: float: simulation timestep (ms)
        :param delay: float: netcon synaptic delay (ms)
        :param v_init: float
        :param verbose: int: level for verbose print statements
        :param debug: bool: turn on for extra tests
        """
        self.pc = pc
        self.delay = delay
        self.tstop = int(tstop)
        self.duration = duration
        self.buffer = buffer
        self.equilibrate = equilibrate
        if dt is None:
            dt = h.dt
        self.dt = dt
        self.v_init = v_init
        self.verbose = verbose
        self.debug = debug

        self.pop_sizes = pop_sizes
        self.total_cells = np.sum(list(self.pop_sizes.values()))

        self.pop_gid_ranges = pop_gid_ranges
        self.pop_cell_types = pop_cell_types
        self.pop_syn_counts = pop_syn_counts
        self.pop_syn_proportions = pop_syn_proportions
        self.connection_weights_mean = connection_weights_mean
        self.connection_weights_norm_sigma = connection_weights_norm_sigma
        self.syn_mech_params = syn_mech_params
        if syn_mech_names is None:
            self.syn_mech_names = default_syn_mech_names
        if syn_mech_param_rules is None:
            self.syn_mech_param_rules = default_syn_mech_param_rules
        if syn_mech_param_defaults is None:
            self.syn_mech_param_defaults = defaultdict(dict)
            for target_pop_name in self.pop_syn_proportions:
                for syn_type in self.pop_syn_proportions[target_pop_name]:
                    if syn_type not in default_syn_type_mech_params:
                        raise RuntimeError('SimpleNetwork: default synaptic mechanism parameters not found for '
                                           'target_pop: %s, syn_type: %s' % (target_pop_name, syn_type))
                    for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]:
                        self.syn_mech_param_defaults[target_pop_name][source_pop_name] = \
                            default_syn_type_mech_params[syn_type]

        self.spike_times_dict = defaultdict(dict)
        self.input_pop_t = dict()
        self.input_pop_firing_rates = defaultdict(dict)

        self.local_random = random.Random()
        self.local_np_random = np.random.RandomState()

        self.cells = defaultdict(dict)
        self.mkcells()
        if self.debug:
            self.verify_cell_types()

        self.ncdict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.voltage_record()
        self.spike_record()

    def mkcells(self):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())

        for pop_name, (gid_start, gid_stop) in viewitems(self.pop_gid_ranges):
            cell_type = self.pop_cell_types[pop_name]
            for i, gid in enumerate(range(gid_start, gid_stop)):
                # round-robin distribution of cells across MPI ranks
                if i % nhost == rank:
                    if self.verbose > 1:
                        print('SimpleNetwork.mkcells: rank: %i got %s gid: %i' % (rank, pop_name, gid))
                    if cell_type == 'input':
                        cell = FFCell(pop_name, gid)
                    elif cell_type == 'minimal':
                        cell = MinimalCell(pop_name, gid)
                    elif cell_type in izhi_cell_type_param_dict:
                        cell = IzhiCell(pop_name, gid, cell_type=cell_type)
                    else:
                        raise RuntimeError('SimpleNetwork.mkcells: %s gid: %i; unrecognized cell type: %s' %
                                           (pop_name, gid, cell_type))
                    self.cells[pop_name][gid] = cell
                    self.pc.set_gid2node(gid, rank)
                    nc = cell.spike_detector
                    self.pc.cell(gid, nc)
        sys.stdout.flush()
        self.pc.barrier()

    def verify_cell_types(self):
        """
        Double-checks each created cell has the intended cell type.
        """
        for pop_name in self.cells:
            target_cell_type = self.pop_cell_types[pop_name]
            for gid in self.cells[pop_name]:
                this_cell = self.cells[pop_name][gid]
                if isinstance(this_cell, FFCell):
                    if target_cell_type != 'input':
                        found_cell_type = type(this_cell)
                        raise RuntimeError('SimpleNetwork.verify_cell_types: %s gid: %i should be FFCell, but '
                                           'is cell_type: %s' % (pop_name, gid, found_cell_type))
                elif isinstance(this_cell, IzhiCell):
                    if target_cell_type != this_cell.cell_type:
                        raise RuntimeError('SimpleNetwork.verify_cell_types: %s gid: %i should be %s, but is '
                                           'IzhiCell type: %s' % (pop_name, gid, target_cell_type, this_cell.cell_type))
                    else:
                        target_izhi_celltype = izhi_cell_type_param_dict[target_cell_type].celltype
                        if target_izhi_celltype != this_cell.izh.celltype:
                            raise RuntimeError('SimpleNetwork.verify_cell_types: %s gid: %i; should be '
                                               'Izhi type %i, but is type %i' %
                                               (pop_name, gid, target_izhi_celltype, this_cell.izh.celltype))
                else:
                    raise RuntimeError('SimpleNetwork.verify_cell_types: %s gid: %i is an unknown type: %s' %
                                       (pop_name, gid, type(this_cell)))

    def set_input_pattern(self, input_types, input_mean_rates=None, input_min_rates=None, input_max_rates=None,
                          input_norm_tuning_widths=None, tuning_peak_locs=None, track_wrap_around=False,
                          spikes_seed=100000000, tuning_duration=None):
        """

        :param input_types: dict
        :param input_mean_rates: dict
        :param input_min_rates: dict
        :param input_max_rates: dict
        :param input_norm_tuning_widths: dict
        :param tuning_peak_locs: dict
        :param track_wrap_around: bool
        :param spikes_seed: int: random seed for reproducible input spike trains
        :param tuning_duration: float
        :param equilibrate: float
        """
        if self.equilibrate > 0.:
            equilibrate_len = int(self.equilibrate/self.dt)
            equilibrate_rate_array = hann(int(self.equilibrate * 2. / self.dt))[:equilibrate_len]
        spikes_seed = int(spikes_seed)
        for pop_name in (pop_name for pop_name in input_types if pop_name in self.cells):
            if input_types[pop_name] == 'constant':
                if input_mean_rates is None or pop_name not in input_mean_rates:
                    raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_mean_rates required to specify '
                                       '%s input population: %s' % (input_types[pop_name], pop_name))
                this_mean_rate = input_mean_rates[pop_name]
                if pop_name not in self.input_pop_t:
                    if self.equilibrate > 0.:
                        self.input_pop_t[pop_name] = \
                            np.append(np.arange(0., self.equilibrate, self.dt), [self.equilibrate, self.tstop])
                    else:
                        self.input_pop_t[pop_name] = [0., self.tstop]
                for gid in self.cells[pop_name]:
                    if self.equilibrate > 0.:
                        self.input_pop_firing_rates[pop_name][gid] = \
                            np.append(this_mean_rate * equilibrate_rate_array, [this_mean_rate, this_mean_rate])
                    else:
                        self.input_pop_firing_rates[pop_name][gid] = [this_mean_rate, this_mean_rate]
            elif input_types[pop_name] == 'gaussian':
                if pop_name not in tuning_peak_locs:
                    raise RuntimeError('SimpleNetwork.set_input_pattern: missing tuning_peak_locs required to specify '
                                       '%s input population: %s' % (input_types[pop_name], pop_name))
                try:
                    this_min_rate = input_min_rates[pop_name]
                    this_max_rate = input_max_rates[pop_name]
                    this_norm_tuning_width = input_norm_tuning_widths[pop_name]
                except Exception:
                    raise RuntimeError('SimpleNetwork.set_input_pattern: missing kwarg(s) required to specify %s input '
                                       'population: %s' % (input_types[pop_name], pop_name))

                this_stim_t = np.arange(0., self.tstop + self.dt / 2., self.dt)
                if pop_name not in self.input_pop_t:
                    self.input_pop_t[pop_name] = this_stim_t

                this_tuning_width = tuning_duration * this_norm_tuning_width
                this_sigma = this_tuning_width / 3. / np.sqrt(2.)
                for gid in self.cells[pop_name]:
                    peak_loc = tuning_peak_locs[pop_name][gid]
                    self.input_pop_firing_rates[pop_name][gid] = \
                        get_gaussian_rate(duration=tuning_duration, peak_loc=peak_loc, sigma=this_sigma,
                                          min_rate=this_min_rate, max_rate=this_max_rate, dt=self.dt,
                                          wrap_around=track_wrap_around, buffer=self.buffer,
                                          equilibrate=self.equilibrate)
                    if self.equilibrate > 0.:
                        self.input_pop_firing_rates[pop_name][gid][:equilibrate_len] *= equilibrate_rate_array

        for pop_name in (pop_name for pop_name in input_types if pop_name in self.cells):
            for gid in self.cells[pop_name]:
                self.local_random.seed(spikes_seed + gid)
                this_spike_train = \
                    get_inhom_poisson_spike_times_by_thinning(self.input_pop_firing_rates[pop_name][gid],
                                                              self.input_pop_t[pop_name], dt=self.dt,
                                                              generator=self.local_random)
                cell = self.cells[pop_name][gid]
                cell.load_vecstim(this_spike_train)

    def set_offline_input_pattern(self, input_types, input_offline_min_rates=None, input_offline_mean_rates=None,
                                  input_offline_fraction_active=None, tuning_peak_locs=None, track_wrap_around=False,
                                  stim_edge_duration=(0., 0.), selection_seed=1000000000, spikes_seed=100000000,
                                  tuning_duration=None):
        """

        :param input_types: dict
        :param input_offline_min_rates: dict
        :param input_offline_mean_rates: dict
        :param input_offline_fraction_active: dict
        :param tuning_peak_locs: dict
        :param track_wrap_around: bool
        :param stim_edge_duration: tuple of float
        :param selection_seed: int: random seed for reproducible input cell selection
        :param spikes_seed: int: random seed for reproducible input spike trains
        :param tuning_duration: float
        """
        if stim_edge_duration[0] > 0.:
            stim_onset_len = int(stim_edge_duration[0]/self.dt)
            stim_onset = hann(int(stim_edge_duration[0] * 2. / self.dt))[:stim_onset_len]
        if stim_edge_duration[1] > 0.:
            stim_offset_len = int(stim_edge_duration[1] / self.dt)
            stim_offset = hann(int(stim_edge_duration[1] * 2. / self.dt))[-stim_offset_len:]
        spikes_seed = int(spikes_seed)
        selection_seed = int(selection_seed)
        self.local_random.seed(selection_seed)
        for pop_name in (pop_name for pop_name in input_types if pop_name in self.cells):
            if input_offline_min_rates is None or pop_name not in input_offline_min_rates:
                raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_offline_min_rates required to '
                                   'specify %s input population: %s' % (input_types[pop_name], pop_name))
            if input_offline_mean_rates is None or pop_name not in input_offline_mean_rates:
                raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_offline_mean_rates required to '
                                   'specify %s input population: %s' % (input_types[pop_name], pop_name))
            if input_offline_fraction_active is None or pop_name not in input_offline_fraction_active:
                raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_offline_fraction_active '
                                   'required to specify %s input population: %s' %
                                   (input_types[pop_name], pop_name))
            if input_types[pop_name] in ['constant', 'gaussian']:
                this_min_rate = input_offline_min_rates[pop_name]
                this_mean_rate = input_offline_mean_rates[pop_name]
                if pop_name not in self.input_pop_t:
                    self.input_pop_t[pop_name] = np.concatenate((
                        [0., self.buffer + self.equilibrate],
                        np.arange(self.buffer + self.equilibrate,
                                  self.buffer + self.equilibrate + stim_edge_duration[0], self.dt),
                        [self.buffer + self.equilibrate + stim_edge_duration[0],
                         self.buffer + self.equilibrate + self.duration - stim_edge_duration[1]],
                        np.arange(self.buffer + self.equilibrate + self.duration - stim_edge_duration[1],
                                  self.buffer + self.equilibrate + self.duration, self.dt),
                        [self.buffer + self.equilibrate + self.duration, self.tstop]))
                this_fraction_active = input_offline_fraction_active[pop_name]
                for gid in self.cells[pop_name]:
                    if self.local_random.random() <= this_fraction_active:
                        components = [[this_min_rate, this_min_rate]]
                        if stim_edge_duration[0] > 0.:
                            components.append((this_mean_rate - this_min_rate) * stim_onset + this_min_rate)
                        components.append([this_mean_rate, this_mean_rate])
                        if stim_edge_duration[1] > 0.:
                            components.append((this_mean_rate - this_min_rate) * stim_offset + this_min_rate)
                        components.append([this_min_rate, this_min_rate])
                        self.input_pop_firing_rates[pop_name][gid] = np.concatenate(components)
                    else:
                        self.input_pop_firing_rates[pop_name][gid] = \
                            np.ones_like(self.input_pop_t[pop_name]) * this_min_rate

        for pop_name in (pop_name for pop_name in input_types if pop_name in self.cells):
            for gid in self.cells[pop_name]:
                self.local_random.seed(spikes_seed + gid)
                this_spike_train = \
                    get_inhom_poisson_spike_times_by_thinning(self.input_pop_firing_rates[pop_name][gid],
                                                              self.input_pop_t[pop_name], dt=self.dt,
                                                              generator=self.local_random)
                cell = self.cells[pop_name][gid]
                cell.load_vecstim(this_spike_train)

    def get_prob_connection_uniform(self, potential_source_gids):
        """

        :param potential_source_gids: array of int
        :return: array of float
        """
        prob_connection = np.ones(len(potential_source_gids), dtype='float32')
        prob_sum = np.sum(prob_connection)
        if prob_sum == 0.:
            return None
        prob_connection /= prob_sum
        return prob_connection

    def get_prob_connection_gaussian(self, potential_source_gids, target_gid, source_pop_name, target_pop_name,
                                     pop_cell_positions, pop_axon_extents):
        """

        :param potential_source_gids: array of int
        :param target_gid: int
        :param source_pop_name: str
        :param target_pop_name: str
        :param pop_cell_positions: tuple of float
        :param pop_axon_extents: float
        :return: array of float
        """
        from scipy.spatial.distance import cdist
        target_cell_position = pop_cell_positions[target_pop_name][target_gid]
        source_cell_positions = \
            [pop_cell_positions[source_pop_name][source_gid] for source_gid in potential_source_gids]
        distances = cdist([target_cell_position], source_cell_positions)[0]
        sigma = pop_axon_extents[source_pop_name] / 3. / np.sqrt(2.)
        prob_connection = np.exp(-(distances / sigma) ** 2.)
        prob_sum = np.sum(prob_connection)
        if prob_sum == 0.:
            return None
        prob_connection /= prob_sum
        return prob_connection

    def connect_cells(self, connectivity_type='uniform', connection_seed=0, **kwargs):
        """

        :param connectivity_type: str
        :param connection_seed: int: random seed for reproducible connections
        """
        connection_seed = int(connection_seed)
        rank = int(self.pc.id())
        for target_pop_name in self.pop_syn_proportions:
            total_syn_count = self.pop_syn_counts[target_pop_name]
            for target_gid in self.cells[target_pop_name]:
                self.local_np_random.seed(connection_seed + target_gid)
                target_cell = self.cells[target_pop_name][target_gid]
                for syn_type in self.pop_syn_proportions[target_pop_name]:
                    for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]:
                        p_syn_count = self.pop_syn_proportions[target_pop_name][syn_type][source_pop_name]
                        this_syn_count = self.local_np_random.binomial(total_syn_count, p_syn_count)
                        potential_source_gids = np.arange(self.pop_gid_ranges[source_pop_name][0],
                                                          self.pop_gid_ranges[source_pop_name][1], 1)
                        # avoid connections to self
                        p_connection = None
                        potential_source_gids = potential_source_gids[potential_source_gids != target_gid]
                        if connectivity_type == 'uniform':
                            p_connection = self.get_prob_connection_uniform(potential_source_gids)
                        elif connectivity_type == 'gaussian':
                            p_connection = \
                                self.get_prob_connection_gaussian(potential_source_gids, target_gid, source_pop_name,
                                                                  target_pop_name, **kwargs)
                        if p_connection is None:
                            continue

                        this_source_gids = self.local_np_random.choice(potential_source_gids, size=this_syn_count,
                                                                       p=p_connection)
                        for source_gid in this_source_gids:
                            this_syn, this_nc = append_connection(
                                target_cell, self.pc, source_pop_name, syn_type, source_gid, delay=self.delay,
                                syn_mech_names=self.syn_mech_names, syn_mech_param_rules=self.syn_mech_param_rules,
                                syn_mech_param_defaults=self.syn_mech_param_defaults[target_pop_name][source_pop_name],
                                **self.syn_mech_params[target_pop_name][source_pop_name])
                            self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid].append(this_nc)
                        if self.verbose > 1:
                            print('SimpleNetwork.connect_cells_%s: rank: %i; target: %s gid: %i; syn_type: %s; '
                                  'source: %s; syn_count: %i' %
                                  (connectivity_type, rank, target_pop_name, target_gid, syn_type, source_pop_name,
                                   this_syn_count))
        sys.stdout.flush()
        self.pc.barrier()

    def assign_connection_weights(self, default_weight_distribution_type='normal',
                                  connection_weight_distribution_types=None, weights_seed=200000000):
        """

        :param default_weight_distribution_type: str
        :param connection_weight_distribution_types: nested dict: {target_pop_name: {source_pop_name: str}}
        :param weights_seed: int: random seed for reproducible connection weights
        """
        rank = int(self.pc.id())
        weights_seed = int(weights_seed)
        for target_pop_name in self.ncdict:
            for target_gid in self.ncdict[target_pop_name]:
                self.local_np_random.seed(weights_seed + target_gid)
                target_cell = self.cells[target_pop_name][target_gid]
                for syn_type in self.pop_syn_proportions[target_pop_name]:
                    for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]:
                        if source_pop_name not in self.ncdict[target_pop_name][target_gid]:
                            continue
                        this_weight_distribution_type = default_weight_distribution_type
                        if connection_weight_distribution_types is not None:
                            if target_pop_name in connection_weight_distribution_types and \
                                    source_pop_name in connection_weight_distribution_types[target_pop_name]:
                                this_weight_distribution_type = \
                                    connection_weight_distribution_types[target_pop_name][source_pop_name]
                        mu = self.connection_weights_mean[target_pop_name][source_pop_name]
                        norm_sigma = self.connection_weights_norm_sigma[target_pop_name][source_pop_name]
                        if self.debug and self.verbose > 1:
                            print('SimpleNetwork.assign_connection_weights: rank: %i, target: %s, source: %s, '
                                  'dist_type: %s, mu: %.3f, norm_sigma: %.3f' %
                                  (rank, target_pop_name, source_pop_name, this_weight_distribution_type, mu,
                                   norm_sigma))

                        for source_gid in self.ncdict[target_pop_name][target_gid][source_pop_name]:
                            if this_weight_distribution_type == 'normal':
                                # enforce weights to be greater than 0
                                this_weight = -1.
                                while this_weight <= 0.:
                                    this_weight = mu * self.local_np_random.normal(1., norm_sigma)
                            elif this_weight_distribution_type == 'lognormal':
                                # enforce weights to be less than 5-fold greater than mean
                                this_weight = 5. * mu
                                while this_weight >= 5. * mu:
                                    this_weight = mu * self.local_np_random.lognormal(0., norm_sigma)
                            else:
                                raise RuntimeError('SimpleNetwork.assign_connection_weights: invalid connection '
                                                   'weight distribution type: %s' % this_weight_distribution_type)
                            this_syn = target_cell.syns[syn_type][source_pop_name]
                            # Assign the same weight to all connections from the same source_gid
                            for this_nc in self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid]:
                                config_connection(syn_type, syn=this_syn, nc=this_nc,
                                                  syn_mech_names=self.syn_mech_names,
                                                  syn_mech_param_rules=self.syn_mech_param_rules,
                                                  weight=this_weight)
        sys.stdout.flush()
        self.pc.barrier()

    def structure_connection_weights(self, structured_weight_params, tuning_peak_locs, wrap_around=True,
                                     tuning_duration=None):
        """

        :param structured_weight_params: nested dict
        :param tuning_peak_locs: nested dict: {'pop_name': {'gid': float} }
        :param wrap_around: bool
        :param tuning_duration: float
        """
        rank = int(self.pc.id())
        for target_pop_name in (target_pop_name for target_pop_name in structured_weight_params
                                if target_pop_name in self.ncdict):
            if target_pop_name not in tuning_peak_locs:
                raise RuntimeError('SimpleNetwork.structure_connection_weights: spatial tuning locations not found for '
                                   'target population: %s' % target_pop_name)
            this_tuning_type = structured_weight_params[target_pop_name]['tuning_type']
            this_peak_delta_weight = structured_weight_params[target_pop_name]['peak_delta_weight']
            this_norm_tuning_width = structured_weight_params[target_pop_name]['norm_tuning_width']
            this_tuning_width = tuning_duration * this_norm_tuning_width
            this_sigma = this_tuning_width / 3. / np.sqrt(2.)
            this_tuning_f = lambda delta_loc: this_peak_delta_weight * np.exp(-(delta_loc / this_sigma) ** 2.)
            for syn_type in self.pop_syn_proportions[target_pop_name]:
                for source_pop_name in \
                        (source_pop_name for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]
                         if source_pop_name in structured_weight_params[target_pop_name]['source_pop_names']):
                    if source_pop_name not in tuning_peak_locs:
                        raise RuntimeError('SimpleNetwork.structure_connection_weights: spatial tuning locations not '
                                           'found for source population: %s' % source_pop_name)
                    for target_gid in (target_gid for target_gid in self.ncdict[target_pop_name]
                                       if source_pop_name in self.ncdict[target_pop_name][target_gid]):
                        target_cell = self.cells[target_pop_name][target_gid]
                        this_syn = target_cell.syns[syn_type][source_pop_name]
                        this_target_loc = tuning_peak_locs[target_pop_name][target_gid]
                        for source_gid in self.ncdict[target_pop_name][target_gid][source_pop_name]:
                            this_delta_loc = abs(tuning_peak_locs[source_pop_name][source_gid] - this_target_loc)
                            if wrap_around:
                                if this_delta_loc > tuning_duration / 2.:
                                    this_delta_loc = tuning_duration - this_delta_loc
                            this_delta_weight = this_tuning_f(this_delta_loc)
                            for this_nc in self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid]:
                                initial_weight = get_connection_param(syn_type, 'weight', syn=this_syn, nc=this_nc,
                                                                      syn_mech_names=self.syn_mech_names,
                                                                      syn_mech_param_rules=self.syn_mech_param_rules)
                                if this_tuning_type == 'additive':
                                    updated_weight = initial_weight + this_delta_weight
                                elif this_tuning_type == 'multiplicative':
                                    updated_weight = initial_weight * (1. + this_delta_weight)
                                if self.debug and self.verbose > 1:
                                    print('SimpleNetwork.structure_connection_weights; rank: %i, target_pop_name: %s, '
                                          'target_gid: %i; source_pop_name: %s, source_gid: %i, initial weight: %.3f, '
                                          'updated weight: %.3f' %
                                          (rank, target_pop_name, target_gid, source_pop_name, source_gid,
                                           initial_weight, updated_weight))
                                config_connection(syn_type, syn=this_syn, nc=this_nc,
                                                  syn_mech_names=self.syn_mech_names,
                                                  syn_mech_param_rules=self.syn_mech_param_rules, weight=updated_weight)
        sys.stdout.flush()
        self.pc.barrier()

    def get_connectivity_dict(self):
        connectivity_dict = dict()
        for target_pop_name in self.pop_syn_proportions:
            connectivity_dict[target_pop_name] = dict()
            for target_gid in self.cells[target_pop_name]:
                connectivity_dict[target_pop_name][target_gid] = dict()
                for syn_type in self.pop_syn_proportions[target_pop_name]:
                    for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]:
                        if source_pop_name in self.ncdict[target_pop_name][target_gid] and \
                                len(self.ncdict[target_pop_name][target_gid][source_pop_name]) > 0:
                            source_gids = []
                            for source_gid in self.ncdict[target_pop_name][target_gid][source_pop_name]:
                                source_gids.extend([source_gid] *
                                                   len(self.ncdict[target_pop_name][target_gid][source_pop_name][
                                                           source_gid]))
                            connectivity_dict[target_pop_name][target_gid][source_pop_name] = source_gids
        return connectivity_dict

    # Instrumentation - stimulation and recording
    def spike_record(self):
        for pop_name in self.cells:
            for gid, cell in viewitems(self.cells[pop_name]):
                tvec = h.Vector()
                nc = cell.spike_detector
                nc.record(tvec)
                self.spike_times_dict[pop_name][gid] = tvec

    def voltage_record(self):
        self.voltage_recvec = defaultdict(dict)
        for pop_name in self.cells:
            for gid, cell in viewitems(self.cells[pop_name]):
                if cell.is_art(): continue
                rec = h.Vector()
                rec.record(getattr(cell.sec(.5), '_ref_v'))
                self.voltage_recvec[pop_name][gid] = rec

    def run(self):
        h.celsius = 35.  # degrees C
        self.pc.set_maxstep(10.)
        h.dt = self.dt
        h.finitialize(self.v_init)
        self.pc.psolve(self.tstop)

    def get_spike_times_dict(self):
        spike_times_dict = dict()
        for pop_name in self.spike_times_dict:
            spike_times_dict[pop_name] = dict()
            for gid, spike_train in viewitems(self.spike_times_dict[pop_name]):
                if len(spike_train) > 0:
                    spike_train_array = np.subtract(np.array(spike_train, dtype='float32'),
                                                    self.buffer + self.equilibrate)
                else:
                    spike_train_array = np.array([], dtype='float32')
                spike_times_dict[pop_name][gid] = spike_train_array
        return spike_times_dict

    def get_voltage_rec_dict(self):
        voltage_rec_dict = dict()
        for pop_name in self.voltage_recvec:
            voltage_rec_dict[pop_name] = dict()
            for gid, recvec in viewitems(self.voltage_recvec[pop_name]):
                voltage_rec_dict[pop_name][gid] = np.array(recvec)
        return voltage_rec_dict

    def get_connection_weights(self):
        weights = dict()
        target_gids = dict()
        for target_pop_name in self.ncdict:
            weights[target_pop_name] = dict()
            target_gids[target_pop_name] = sorted(self.ncdict[target_pop_name].keys())
            num_cells_target_pop = len(target_gids[target_pop_name])
            for syn_type in self.pop_syn_proportions[target_pop_name]:
                for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]:
                    weights[target_pop_name][source_pop_name] = \
                        np.zeros([num_cells_target_pop, self.pop_sizes[source_pop_name]])
                    for i, target_gid in enumerate(target_gids[target_pop_name]):
                        target_cell = self.cells[target_pop_name][target_gid]
                        if syn_type in target_cell.syns and source_pop_name in target_cell.syns[syn_type]:
                            this_syn = target_cell.syns[syn_type][source_pop_name]
                        else:
                            this_syn = None
                        start_gid = self.pop_gid_ranges[source_pop_name][0]
                        stop_gid = self.pop_gid_ranges[source_pop_name][1]
                        for j, source_gid in enumerate(range(start_gid, stop_gid)):
                            if source_gid in self.ncdict[target_pop_name][target_gid][source_pop_name]:
                                this_weight_list = []
                                for this_nc in self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid]:
                                    this_weight_list.append(
                                        get_connection_param(syn_type, 'weight', syn=this_syn, nc=this_nc,
                                                             syn_mech_names=self.syn_mech_names,
                                                             syn_mech_param_rules=self.syn_mech_param_rules))
                                this_weight = np.mean(this_weight_list)
                                weights[target_pop_name][source_pop_name][i][j] = this_weight

        return target_gids, weights


def get_connection_param(syn_type, syn_mech_param, syn=None, nc=None, delay=None, syn_mech_names=None,
                         syn_mech_param_rules=None):
    """
    :param syn_type: str
    :param syn_mech_param: str
    :param syn: NEURON point process object
    :param nc: NEURON netcon object
    :param delay: float
    :param syn_mech_names: dict
    :param syn_mech_param_rules: dict
    """
    if syn is None and nc is None:
        raise RuntimeError('get_connection_param: must provide at least one: synaptic point process or netcon object')
    if nc is not None and delay is not None:
        nc.delay = delay
    if syn_mech_names is None:
        syn_mech_names = default_syn_mech_names
    syn_mech_name = syn_mech_names[syn_type]
    if syn_mech_param_rules is None:
        syn_mech_param_rules = default_syn_mech_param_rules
    if syn_mech_param in syn_mech_param_rules[syn_mech_name]['mech_params'] and syn is not None:
        return getattr(syn, syn_mech_param)
    elif syn_mech_param in syn_mech_param_rules[syn_mech_name]['netcon_params'] and nc is not None:
        index = syn_mech_param_rules[syn_mech_name]['netcon_params'][syn_mech_param]
        return nc.weight[index]
    else:
        raise RuntimeError('get_connection_param: invalid syn_mech_param: %s' % syn_mech_param)


def config_connection(syn_type, syn=None, nc=None, delay=None, syn_mech_names=None, syn_mech_param_rules=None,
                      **syn_mech_params):
    """
    :param syn_type: str
    :param syn: NEURON point process object
    :param nc: NEURON netcon object
    :param delay: float
    :param syn_mech_names: dict
    :param syn_mech_param_rules: dict
    :param syn_mech_params: dict
    """
    if syn is None and nc is None:
        raise RuntimeError('config_connection: must provide at least one: synaptic point process or netcon object')
    if nc is not None and delay is not None:
        nc.delay = delay
    if syn_mech_names is None:
        syn_mech_names = default_syn_mech_names
    syn_mech_name = syn_mech_names[syn_type]
    if syn_mech_param_rules is None:
        syn_mech_param_rules = default_syn_mech_param_rules
    for param_name in syn_mech_params:
        if param_name in syn_mech_param_rules[syn_mech_name]['mech_params'] and syn is not None:
            setattr(syn, param_name, syn_mech_params[param_name])
        elif param_name in syn_mech_param_rules[syn_mech_name]['netcon_params'] and nc is not None:
            index = syn_mech_param_rules[syn_mech_name]['netcon_params'][param_name]
            nc.weight[index] = syn_mech_params[param_name]


def append_connection(cell, pc, source_pop_name, syn_type, source_gid, delay=None, syn_mech_names=None,
                      syn_mech_param_rules=None, syn_mech_param_defaults=None, **kwargs):
    """

    :param cell: e.g. :class:'IzhiCell', :class:'MinimalCell'
    :param pc: :class:'h.ParallelContext'
    :param source_pop_name: str
    :param syn_type: str
    :param source_gid: int
    :param delay: float
    :param syn_mech_names: dict
    :param syn_mech_param_rules: nested dict
    :param syn_mech_param_defaults: nested dict
    :param kwargs: dict

    """
    if syn_mech_names is None:
        syn_mech_names = default_syn_mech_names
    syn_mech_name = syn_mech_names[syn_type]
    if syn_mech_param_defaults is None:
        syn_mech_params = dict(default_syn_type_mech_params[syn_type])
    else:
        syn_mech_params = dict(syn_mech_param_defaults)
    syn_mech_params.update(kwargs)
    if syn_type in cell.syns and source_pop_name in cell.syns[syn_type]:
        syn = cell.syns[syn_type][source_pop_name]
    else:
        syn = getattr(h, syn_mech_name)(cell.sec(0.5))
        cell.syns[syn_type][source_pop_name] = syn

    nc = pc.gid_connect(source_gid, syn)
    config_connection(syn_type, syn=syn, nc=nc, delay=delay, syn_mech_names=syn_mech_names,
                      syn_mech_param_rules=syn_mech_param_rules, **syn_mech_params)

    return syn, nc


class IzhiCell(object):
    # Integrate-and-fire-like neuronal cell models with additional tunable dynamic parameters (e.g. adaptation).
    # Derived from http://modeldb.yale.edu/39948
    def __init__(self, pop_name=None, gid=None, cell_type='RS'):
        """

        :param pop_name: str
        :param gid: int
        :param cell_type: str
        """
        self.cell_type = cell_type
        self.sec = h.Section(cell=self)
        self.sec.L, self.sec.diam = 10., 10.
        self.izh = h.Izhi2019(.5, sec=self.sec)
        self.base_cm = 31.831  # Produces membrane time constant of 8 ms for a RS cell with izh.C = 1. and izi.k = 0.7
        if pop_name is None:
            pop_name = self.cell_type
        self.pop_name = pop_name
        if gid is None:
            gid = 0
        self.gid = gid
        self.name = '%s%s' % (pop_name, gid)

        if self.cell_type not in izhi_cell_type_param_dict:
            raise ValueError('IzhiCell: cell_type: %s not recognized' % cell_type)

        for cell_type_param in izhi_cell_type_param_names:
            setattr(self.izh, cell_type_param, getattr(izhi_cell_type_param_dict[self.cell_type], cell_type_param))

        self.sec.cm = self.base_cm * self.izh.C
        self.syns = defaultdict(dict)
        self.spike_detector = self.connect2target()

    def connect2target(self, target=None):
        nc = h.NetCon(self.sec(1)._ref_v, target, sec=self.sec)
        nc.threshold = izhi_cell_type_param_dict[self.cell_type].vpeak - 1.
        return nc

    def is_art(self):
        return 0


class MinimalCell(object):
    def __init__(self, pop_name, gid):
        """

        :param pop_name: str
        :param gid: int
        """
        self.pop_name = pop_name
        self.gid = gid
        self.name = '%s%s' % (pop_name, gid)
        self.sec = h.Section(cell=self)
        self.sec.L, self.sec.diam = 10., 10.
        self.sec.cm = 31.831
        self.sec.insert('pas')
        self.sec(0.5).e_pas = -65.  # mv
        self.syns = defaultdict(dict)
        self.spike_detector = self.connect2target()

    def connect2target(self, target=None):
        nc = h.NetCon(self.sec(1)._ref_v, target, sec=self.sec)
        # nc.threshold = izhi_cell_type_param_dict[self.cell_type].vpeak
        return nc

    def is_art(self):
        return 0


class FFCell(object):
    def __init__(self, pop_name, gid):
        """

        :param pop_name: str
        :param gid: int
        """
        self.pop_name = pop_name
        self.gid = gid
        self.name = '%s%s' % (pop_name, gid)
        self.vs = h.VecStim()
        self.spike_detector = self.connect2target()
        self.spike_train = []

    def connect2target(self, target=None):
        nc = h.NetCon(self.vs, target)
        return nc

    def load_vecstim(self, spike_train):
        self.spike_train = spike_train
        self.vs.play(h.Vector(spike_train))

    def is_art(self):
        return 1


def check_voltages_exceed_threshold(voltage_rec_dict, input_t, pop_cell_types, valid_t=None):
    """

    :param voltage_rec_dict: nested dict
    :param input_t: array
    :param pop_cell_types: dict of str
    :param valid_t: array
    :return: bool
    """
    if valid_t is None:
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    for pop_name in voltage_rec_dict:
        cell_type = pop_cell_types[pop_name]
        if cell_type not in izhi_cell_types:
            continue
        vt = izhi_cell_type_param_dict[cell_type].vt
        for gid in voltage_rec_dict[pop_name]:
            if np.mean(voltage_rec_dict[pop_name][gid][valid_indexes]) > vt:
                return True
    return False


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


def infer_firing_rates_from_spike_count(binned_spike_count_dict, input_t, output_t, align_to_t=0., window_dur=500.,
                                        step_dur=1., smooth_dur=100., debug=False):
    """

    :param binned_spike_count_dict: dict: {pop_name: {gid: array} }
    :param input_t: array
    :param output_t: array
    :param align_to_t: float
    :param window_dur: float
    :param step_dur: float
    :param smooth_dur: float
    :param debug: bool
    :return: tuple of dict
    """
    dt = output_t[1] - output_t[0]
    half_window_bins = int(window_dur // dt // 2)
    window_bins = int(2 * half_window_bins + 1)
    window_dur = window_bins * dt
    step_bins = step_dur // dt
    smooth_bins = int(smooth_dur // dt)
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    # if possible, include a bin centered on output_t[0]
    binned_t_center_indexes = []
    this_center_index = np.where(output_t >= align_to_t)[0]
    if len(this_center_index) > 0:
        this_center_index = this_center_index[0]
        if this_center_index < half_window_bins:
            this_center_index = half_window_bins
            binned_t_center_indexes.append(this_center_index)
        else:
            while this_center_index > half_window_bins:
                binned_t_center_indexes.append(this_center_index)
                this_center_index -= step_bins
            binned_t_center_indexes.reverse()
    else:
        this_center_index = half_window_bins
        binned_t_center_indexes.append(this_center_index)
    this_center_index = binned_t_center_indexes[-1] + step_bins
    while this_center_index < len(output_t) - half_window_bins:
        binned_t_center_indexes.append(this_center_index)
        this_center_index += step_bins
    binned_t_center_indexes = np.array(binned_t_center_indexes, dtype='int')
    binned_t = output_t[binned_t_center_indexes]

    valid_indexes = np.where((input_t >= output_t[0]) & (input_t <= output_t[-1]))[0]
    firing_rates_from_spike_count_dict = dict()
    plot_count = 0
    for pop_name in binned_spike_count_dict:
        firing_rates_from_spike_count_dict[pop_name] = dict()
        for gid in binned_spike_count_dict[pop_name]:
            this_binned_spike_count = binned_spike_count_dict[pop_name][gid][valid_indexes]
            this_inferred_rate = np.empty_like(binned_t)
            for rate_index, t_center_index in enumerate(binned_t_center_indexes):
                t_start_index = t_center_index - half_window_bins
                t_end_index = t_center_index + half_window_bins + 1
                this_inferred_rate[rate_index] = \
                    np.sum(this_binned_spike_count[t_start_index:t_end_index]) / (window_dur / 1000.)
            this_interp_rate = np.interp(output_t, binned_t, this_inferred_rate)
            this_smoothed_rate = savgol_filter(this_interp_rate, smooth_bins, 3, mode='interp')
            this_smoothed_rate = np.maximum(0., this_smoothed_rate)
            firing_rates_from_spike_count_dict[pop_name][gid] = this_smoothed_rate
            if debug and pop_name == 'FF' and plot_count < 10:
                plot_count += 1
                fig = plt.figure()
                active_indexes = np.where(this_binned_spike_count > 0.)[0]
                plt.plot(input_t[active_indexes], np.ones_like(active_indexes), '.')
                plt.plot(binned_t, this_inferred_rate)
                plt.plot(output_t, this_interp_rate)
                plt.plot(output_t, this_smoothed_rate)
                fig.show()

    return firing_rates_from_spike_count_dict


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
    if pad_len > 0:
        padded_spike_times = valid_spike_times
        r_pad_indexes = np.where((spike_times > t[0]) & (spike_times <= t[pad_len]))[0]
        l_pad_indexes = np.where((spike_times >= t[-pad_len]) & (spike_times < t[-1]))[0]
        if wrap_around:
            if len(r_pad_indexes) > 0:
                r_pad_spike_times = np.add(t[-1]+dt, np.subtract(spike_times[r_pad_indexes], t[0]))
                padded_spike_times = np.append(padded_spike_times, r_pad_spike_times)
            if len(l_pad_indexes) > 0:
                l_pad_spike_times = np.add(t[0], np.subtract(spike_times[l_pad_indexes], t[-1]+dt))
                padded_spike_times = np.append(l_pad_spike_times, padded_spike_times)
        else:
            if len(r_pad_indexes) > 0:
                r_pad_spike_times = np.add(t[0], np.subtract(t[0], spike_times[r_pad_indexes])[::-1])
                padded_spike_times = np.append(r_pad_spike_times, padded_spike_times)
            if len(l_pad_indexes) > 0:
                l_pad_spike_times = np.add(t[-1]+dt, np.subtract(t[-1]+dt, spike_times[l_pad_indexes])[::-1])
                padded_spike_times = np.append(padded_spike_times, l_pad_spike_times)
        padded_t = \
            np.concatenate((np.arange(-pad_dur, 0., dt), t, np.arange(t[-1] + dt, t[-1] + pad_dur + dt / 2., dt)))
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


def get_binned_spike_count(spike_times, t):
    """
    Convert spike times to a binned binary spike train
    :param spike_times: array (ms)
    :param t: array (ms)
    :return: array
    """
    binned_spikes = np.zeros_like(t)
    if len(spike_times) > 0:
        try:
            spike_indexes = [np.where(t >= spike_time)[0][0] for spike_time in spike_times]
        except Exception as e:
            print(spike_times)
            print(t)
            sys.stdout.flush()
            time.sleep(0.1)
            raise(e)
        binned_spikes[spike_indexes] = 1.
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


def get_pop_activity_stats(firing_rates_dict, input_t, valid_t=None, threshold=2., plot=False):
    """
    Calculate firing rate statistics for each cell population.
    :param firing_rates_dict: nested dict of array
    :param input_t: array
    :param valid_t: array
    :param threshold: firing rate threshold for "active" cells: float (Hz)
    :param plot: bool
    :return: tuple of dict
    """
    min_rate_dict = defaultdict(dict)
    peak_rate_dict = defaultdict(dict)
    mean_rate_active_cells_dict = dict()
    pop_fraction_active_dict = dict()
    mean_min_rate_dict = dict()
    mean_peak_rate_dict = dict()

    if valid_t is None:
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

    if plot:
        fig, axes = plt.subplots(1, 2)
        for pop_name in pop_fraction_active_dict:
            axes[0].plot(valid_t, pop_fraction_active_dict[pop_name], label=pop_name)
            axes[0].set_title('Active fraction of population', fontsize=mpl.rcParams['font.size'])
            axes[1].plot(valid_t, mean_rate_active_cells_dict[pop_name])
            axes[1].set_title('Mean firing rate of active cells', fontsize=mpl.rcParams['font.size'])
        axes[0].set_ylim(0., axes[0].get_ylim()[1])
        axes[1].set_ylim(0., axes[1].get_ylim()[1])
        axes[0].legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    return mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict


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


def get_bandpass_filtered_signal_stats(signal, input_t, sos, filter_band, buffered_sos=None, buffered_filter_band=None,
                                       output_t=None, bins=100, signal_label='', filter_label='',
                                       axis_label='Amplitude', units='a.u.', pad=True, pad_len=None, plot=False,
                                       verbose=False):
    """

    :param signal: array
    :param input_t: array (ms)
    :param sos: array
    :param filter_band: list of float (Hz)
    :param buffered_sos: array
    :param buffered_filter_band: list of float (Hz)
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
            print('%s\n%s bandpass filter (%.1f:%.1f Hz); Failed - no signal' %
                  (signal_label, filter_label, min(filter_band), max(filter_band)))
            sys.stdout.flush()
        return signal, np.zeros_like(signal), 0., 0., 0.
    dt = input_t[1] - input_t[0]  # ms
    fs = 1000. / dt

    if buffered_filter_band is not None:
        if buffered_sos is None:
            raise RuntimeError('get_bandpass_filtered_signal_stats: when specifying a buffered_filter_band, a'
                               'buffered filter must also be provided')
        bandwidth = buffered_filter_band[1] - buffered_filter_band[0]
    else:
        bandwidth = filter_band[1] - filter_band[0]
    nfft = int(fs * bins / bandwidth)

    if pad and pad_len is None:
        pad_dur = min(10. * 1000. / np.min(filter_band), len(input_t) * dt)  # ms
        pad_len = min(int(pad_dur / dt), len(input_t) - 1)
    if pad:
        padded_signal = get_mirror_padded_signal(signal, pad_len)
    else:
        padded_signal = np.array(signal)

    filtered_padded_signal = sosfiltfilt(sos, padded_signal)
    filtered_signal = filtered_padded_signal[pad_len:-pad_len]
    padded_envelope = np.abs(hilbert(filtered_padded_signal))
    envelope = padded_envelope[pad_len:-pad_len]

    if buffered_sos is not None:
        if buffered_filter_band is None:
            raise RuntimeError('get_bandpass_filtered_signal_stats: when specifying a buffered filter, a'
                               'buffered_filter_band must also be provided')
        buffered_filtered_padded_signal = sosfiltfilt(buffered_sos, padded_signal)
        buffered_filtered_signal = buffered_filtered_padded_signal[pad_len:-pad_len]
        f, power = periodogram(buffered_filtered_signal, fs=fs, nfft=nfft)
    else:
        f, power = periodogram(filtered_signal, fs=fs, nfft=nfft)

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
        if buffered_filter_band is not None:
            axes[1][0].set_xlim(min(buffered_filter_band), max(buffered_filter_band))
        else:
            axes[1][0].set_xlim(min(filter_band), max(filter_band))

        clean_axes(axes)
        fig.suptitle('%s: %s bandpass filter (%.1f:%.1f Hz)\nEnvelope ratio: %.3f; Centroid freq: %.3f Hz\n'
                     'Frequency tuning index: %.3f' % (signal_label, filter_label, min(filter_band), max(filter_band),
                                                       envelope_ratio, centroid_freq, freq_tuning_index),
                     fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.75, hspace=0.3)
        fig.show()

    return filtered_signal, envelope, envelope_ratio, centroid_freq, freq_tuning_index


def get_pop_bandpass_filtered_signal_stats(signal_dict, filter_band_dict, input_t, valid_t=None, output_t=None,
                                           order=15, plot=False, verbose=False):
    """

    :param signal_dict: array
    :param filter_band_dict: dict: {filter_label (str): list of float (Hz) }
    :param input_t: array (ms)
    :param valid_t: array (ms)
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

    if valid_t is None:
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    for filter_label, filter_band in viewitems(filter_band_dict):
        filtered_signal_dict[filter_label] = {}
        envelope_dict[filter_label] = {}
        envelope_ratio_dict[filter_label] = {}
        centroid_freq_dict[filter_label] = {}
        freq_tuning_index_dict[filter_label] = {}
        sos = get_butter_bandpass_filter(filter_band, sampling_rate, filter_label=filter_label, order=order, plot=plot)
        buffered_filter_band = [filter_band[0] / 2., 2. * filter_band[1]]
        buffered_sos = get_butter_bandpass_filter(buffered_filter_band, sampling_rate, filter_label=filter_label,
                                                  order=order, plot=False)
        for pop_name in signal_dict:
            signal = signal_dict[pop_name][valid_indexes]
            filtered_signal_dict[filter_label][pop_name], envelope_dict[filter_label][pop_name], \
            envelope_ratio_dict[filter_label][pop_name], centroid_freq_dict[filter_label][pop_name], \
            freq_tuning_index_dict[filter_label][pop_name] = \
                get_bandpass_filtered_signal_stats(signal, valid_t, sos, filter_band, buffered_sos=buffered_sos,
                                                   buffered_filter_band=buffered_filter_band, output_t=output_t,
                                                   signal_label='Population: %s' % pop_name, filter_label=filter_label,
                                                   axis_label='Firing rate', units='Hz', plot=plot, verbose=verbose)

    return filtered_signal_dict, envelope_dict, envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict


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
                             cbar_kw={}, cbar_label="", **kwargs):
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
        pop_names = list(spike_times_dict.keys())
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


def plot_voltage_traces(voltage_rec_dict, input_t, valid_t=None, spike_times_dict=None, rows=3, cols=4, pop_names=None):
    """

    :param voltage_rec_dict: dict of array
    :param input_t: array
    :param valid_t: array
    :param spike_times_dict: nested dict of array
    :param cells_per_pop: int
    :param pop_names: list of str
    """
    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    if pop_names is None:
        pop_names = list(voltage_rec_dict.keys())
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
            axes[row][col].plot(valid_t, rec, label='Vm', c='grey')
            if spike_times_dict is not None and pop_name in spike_times_dict and gid in spike_times_dict[pop_name]:
                binned_spike_indexes = find_nearest(spike_times_dict[pop_name][gid], valid_t)
                axes[row][col].plot(valid_t[binned_spike_indexes], rec[binned_spike_indexes], 'k.', label='Spikes')
            axes[row][col].set_title('gid: {}'.format(gid), fontsize=mpl.rcParams['font.size'])
        axes[0][cols-1].legend(loc='center left', frameon=False, framealpha=0.5, bbox_to_anchor=(1., 0.5),
                               fontsize=mpl.rcParams['font.size'])
        clean_axes(axes)
        fig.suptitle('Voltage recordings: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.85)
        fig.show()


def plot_weight_matrix(connection_weights_dict, pop_gid_ranges, tuning_peak_locs=None, pop_names=None):
    """
    Plots heat maps of connection strengths across all connected cell populations. If input activity or input weights
    are spatially tuned, cell ids are also sorted by peak location.
    :param connection_weights_dict: nested dict: {'target_pop_name': {'target_gid': {'source_pop_name':
                                                    {'source_gid': float} } } }
    :param pop_gid_ranges: dict: {'pop_name', tuple of int}
    :param tuning_peak_locs: nested dict: {'pop_name': {'gid': float} }
    :param pop_names: list of str
    """

    if pop_names is None:
        pop_names = list(connection_weights_dict.keys())
    sorted_gid_indexes = dict()
    for target_pop_name in pop_names:
        if target_pop_name not in connection_weights_dict:
            raise RuntimeError('plot_weight_matrix: missing population: %s' % target_pop_name)
        if target_pop_name not in sorted_gid_indexes and target_pop_name in tuning_peak_locs and \
                len(tuning_peak_locs[target_pop_name]) > 0:
            this_ordered_peak_locs = \
                np.array([tuning_peak_locs[target_pop_name][gid] for gid in range(*pop_gid_ranges[target_pop_name])])
            sorted_gid_indexes[target_pop_name] = np.argsort(this_ordered_peak_locs)
        target_pop_size = pop_gid_ranges[target_pop_name][1] - pop_gid_ranges[target_pop_name][0]
        cols = len(connection_weights_dict[target_pop_name])
        fig, axes = plt.subplots(1, cols, sharey=True, figsize=(5 * cols, 5))
        y_interval = max(2, target_pop_size // 10)
        yticks = list(range(0, target_pop_size, y_interval))
        if target_pop_name in sorted_gid_indexes:
            axes[0].set_ylabel('Target: %s\nSorted Cell ID' % target_pop_name)
            ylabels = yticks
        else:
            axes[0].set_ylabel('Target: %s\nCell ID' % target_pop_name)
            ylabels = np.add(yticks, pop_gid_ranges[target_pop_name][0])

        for col, source_pop_name in enumerate(connection_weights_dict[target_pop_name]):
            if source_pop_name not in sorted_gid_indexes and source_pop_name in tuning_peak_locs and \
                    len(tuning_peak_locs[source_pop_name]) > 0:
                this_ordered_peak_locs = \
                    np.array([tuning_peak_locs[source_pop_name][gid]
                              for gid in range(*pop_gid_ranges[source_pop_name])])
                sorted_gid_indexes[source_pop_name] = np.argsort(this_ordered_peak_locs)
            weight_matrix = connection_weights_dict[target_pop_name][source_pop_name]
            if target_pop_name in sorted_gid_indexes:
                weight_matrix[:] = weight_matrix[sorted_gid_indexes[target_pop_name], :]
            if source_pop_name in sorted_gid_indexes:
                weight_matrix[:] = weight_matrix[:, sorted_gid_indexes[source_pop_name]]

            source_pop_size = pop_gid_ranges[source_pop_name][1] - pop_gid_ranges[source_pop_name][0]
            x_interval = max(2, source_pop_size // 10)
            xticks = list(range(0, source_pop_size, x_interval))
            if source_pop_name in sorted_gid_indexes:
                xlabels = xticks
                axes[col].set_xlabel('Sorted Cell ID\nSource: %s' % source_pop_name)
            else:
                xlabels = np.add(xticks, pop_gid_ranges[source_pop_name][0])
                axes[col].set_xlabel('Cell ID\nSource: %s' % source_pop_name)

            plot_heatmap_from_matrix(weight_matrix, xticks=xticks, xtick_labels=xlabels, yticks=yticks,
                                     ytick_labels=ylabels, ax=axes[col], aspect='auto', cbar_label='Synaptic weight',
                                     vmin=0.)
        clean_axes(axes)
        fig.suptitle('Connection weights onto %s population' % target_pop_name, fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, wspace=0.2)
        fig.show()


def plot_firing_rate_heatmaps(firing_rates_dict, input_t, valid_t=None, pop_names=None, tuning_peak_locs=None):
    """

    :param firing_rates_dict: dict of array
    :param input_t: array
    :param valid_t: array
    :param pop_names: list of str
    """
    if valid_t is None:
        valid_t = input_t
        valid_indexes = ()
    else:
        valid_indexes = np.where((input_t >= valid_t[0]) & (input_t <= valid_t[-1]))[0]
    if pop_names is None:
        pop_names = list(firing_rates_dict.keys())
    for pop_name in pop_names:
        sort = pop_name in tuning_peak_locs and len(tuning_peak_locs[pop_name]) > 0
        if sort:
            sorted_indexes = np.argsort(list(tuning_peak_locs[pop_name].values()))
            sorted_gids = np.array(list(tuning_peak_locs[pop_name].keys()))[sorted_indexes]
        else:
            sorted_gids = sorted(list(firing_rates_dict[pop_name].keys()))
        fig, axes = plt.subplots()
        rate_matrix = np.empty((len(sorted_gids), len(valid_t)), dtype='float32')
        for i, gid in enumerate(sorted_gids):
            rate_matrix[i][:] = firing_rates_dict[pop_name][gid][valid_indexes]
        y_interval = max(2, len(sorted_gids) // 10)
        yticks = list(range(0, len(sorted_gids), y_interval))
        ylabels = np.array(sorted_gids)[yticks]
        dt = valid_t[1] - valid_t[0]
        x_interval = min(int(1000. / dt), int((valid_t[-1] + dt) / 5.))
        xticks = list(range(0, len(valid_t), x_interval))
        xlabels = np.array(valid_t)[xticks].astype('int32')
        plot_heatmap_from_matrix(rate_matrix, xticks=xticks, xtick_labels=xlabels, yticks=yticks,
                                 ytick_labels=ylabels, ax=axes, aspect='auto', cbar_label='Firing rate (Hz)',
                                 vmin=0.)
        axes.set_xlabel('Time (ms)')
        axes.set_ylabel('Target: %s\nCell ID' % pop_name)
        if sort:
            axes.set_title('Sorted firing rate: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
        else:
            axes.set_title('Firing rate: %s population' % pop_name, fontsize=mpl.rcParams['font.size'])
        clean_axes(axes)
        fig.tight_layout()
        fig.show()


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


def analyze_simple_network_results_from_file(data_file_path, model_label=None, verbose=False, return_context=False,
                                             plot=True):
    """

    :param data_file_path: str (path)
    :param model_label: int or str
    :param verbose: bool
    :param return_context: bool
    :param plot: bool
    """
    if not os.path.isfile(data_file_path):
        raise IOError('analyze_simple_network_results_from_file: invalid data file path: %s' % data_file_path)

    full_spike_times_dict = defaultdict(dict)
    buffered_firing_rates_dict = defaultdict(dict)
    buffered_firing_rates_from_binned_spike_count_dict = defaultdict(dict)
    full_binned_spike_count_dict = defaultdict(dict)
    filter_bands = dict()
    subset_full_voltage_rec_dict = defaultdict(dict)
    connection_weights_dict = dict()
    tuning_peak_locs = dict()
    connectivity_dict = dict()
    pop_syn_proportions = dict()
    pop_cell_positions = dict()

    exported_data_key = 'simple_network_exported_data'
    with h5py.File(data_file_path, 'r') as f:
        group = get_h5py_group(f, [model_label, exported_data_key])
        connectivity_type = get_h5py_attr(group.attrs, 'connectivity_type')
        active_rate_threshold = group.attrs['active_rate_threshold']
        pop_gid_ranges = dict()
        for pop_name in group['pop_gid_ranges']:
            pop_gid_ranges[pop_name] = tuple(group['pop_gid_ranges'][pop_name][:])
        subgroup = group['full_spike_times']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        subgroup = group['buffered_firing_rates']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                buffered_firing_rates_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        subgroup = group['buffered_firing_rates_from_binned_spike_count']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                buffered_firing_rates_from_binned_spike_count_dict[pop_name][int(gid_key)] = \
                    subgroup[pop_name][gid_key][:]
        subgroup = group['full_binned_spike_count']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                full_binned_spike_count_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        full_binned_t = group['full_binned_t'][:]
        buffered_binned_t = group['buffered_binned_t'][:]
        binned_t = group['binned_t'][:]
        subgroup = group['filter_bands']
        for filter in subgroup:
            filter_bands[filter] = subgroup[filter][:]
        subgroup = group['subset_full_voltage_recs']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                subset_full_voltage_rec_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        full_rec_t = group['full_rec_t'][:]
        buffered_rec_t = group['buffered_rec_t'][:]
        rec_t = group['rec_t'][:]
        subgroup = group['connection_weights']
        for target_pop_name in subgroup:
            connection_weights_dict[target_pop_name] = dict()
            for source_pop_name in subgroup[target_pop_name]:
                connection_weights_dict[target_pop_name][source_pop_name] = \
                    subgroup[target_pop_name][source_pop_name][:]
        if 'tuning_peak_locs' in group and len(group['tuning_peak_locs']) > 0:
            subgroup = group['tuning_peak_locs']
            for pop_name in subgroup:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(subgroup[pop_name]['target_gids'], subgroup[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        subgroup = group['connectivity']
        for target_pop_name in subgroup:
            connectivity_dict[target_pop_name] = dict()
            for target_gid_key in subgroup[target_pop_name]:
                target_gid = int(target_gid_key)
                connectivity_dict[target_pop_name][target_gid] = dict()
                for source_pop_name in subgroup[target_pop_name][target_gid_key]:
                    connectivity_dict[target_pop_name][target_gid][source_pop_name] = \
                        subgroup[target_pop_name][target_gid_key][source_pop_name][:]
        subgroup = group['pop_syn_proportions']
        for target_pop_name in subgroup:
            pop_syn_proportions[target_pop_name] = dict()
            for syn_type in subgroup[target_pop_name]:
                pop_syn_proportions[target_pop_name][syn_type] = dict()
                source_pop_names = subgroup[target_pop_name][syn_type]['source_pop_names'][:].astype('str')
                for source_pop_name, syn_proportion in zip(source_pop_names,
                                                           subgroup[target_pop_name][syn_type]['syn_proportions'][:]):
                    pop_syn_proportions[target_pop_name][syn_type][source_pop_name] = syn_proportion
        subgroup = group['pop_cell_positions']
        for pop_name in subgroup:
            pop_cell_positions[pop_name] = dict()
            for gid, position in zip(subgroup[pop_name]['gids'][:], subgroup[pop_name]['positions'][:]):
                pop_cell_positions[pop_name][gid] = position

    binned_dt = binned_t[1] - binned_t[0]
    full_pop_mean_rate_from_binned_spike_count_dict = \
        get_pop_mean_rate_from_binned_spike_count(full_binned_spike_count_dict, dt=binned_dt)
    _ = get_pop_activity_stats(buffered_firing_rates_dict, input_t=buffered_binned_t, valid_t=binned_t,
                               threshold=active_rate_threshold, plot=plot)
    _ = get_pop_bandpass_filtered_signal_stats(full_pop_mean_rate_from_binned_spike_count_dict, filter_bands,
                                               input_t=full_binned_t, valid_t=buffered_binned_t, output_t=binned_t,
                                               plot=plot, verbose=verbose)
    if plot:
        plot_inferred_spike_rates(full_spike_times_dict, buffered_firing_rates_dict, input_t=buffered_binned_t,
                                  valid_t=binned_t, active_rate_threshold=active_rate_threshold)
        plot_voltage_traces(subset_full_voltage_rec_dict, full_rec_t, valid_t=rec_t,
                            spike_times_dict=full_spike_times_dict)
        plot_weight_matrix(connection_weights_dict, pop_gid_ranges=pop_gid_ranges, tuning_peak_locs=tuning_peak_locs)
        plot_firing_rate_heatmaps(buffered_firing_rates_dict, input_t=buffered_binned_t, valid_t=binned_t,
                                  tuning_peak_locs=tuning_peak_locs)
        if connectivity_type == 'gaussian':
            plot_2D_connection_distance(pop_syn_proportions, pop_cell_positions, connectivity_dict)

    if return_context:
        context = Context()
        context.update(locals())
        return context


def analyze_simple_network_replay_results_from_file(data_file_path, model_label=None, verbose=False, return_context=False,
                                                 plot=True):
    """

    :param data_file_path: str (path)
    :param model_label: int or str
    :param verbose: bool
    :param return_context: bool
    :param plot: bool
    """
    if not os.path.isfile(data_file_path):
        raise IOError('analyze_simple_network_replay_results_from_file: invalid data file path: %s' % data_file_path)

    full_spike_times_dict = defaultdict(dict)
    buffered_firing_rates_dict = defaultdict(dict)
    buffered_firing_rates_from_binned_spike_count_dict = defaultdict(dict)
    full_binned_spike_count_dict = defaultdict(dict)
    filter_bands = dict()
    subset_full_voltage_rec_dict = defaultdict(dict)
    connection_weights_dict = dict()
    tuning_peak_locs = dict()
    connectivity_dict = dict()
    pop_syn_proportions = dict()
    pop_cell_positions = dict()

    exported_data_key = 'simple_network_exported_data'
    with h5py.File(data_file_path, 'r') as f:
        group = get_h5py_group(f, ['shared_context'])
        connectivity_type = get_h5py_attr(group.attrs, 'connectivity_type')
        active_rate_threshold = group.attrs['active_rate_threshold']
        pop_gid_ranges = dict()
        for pop_name in group['pop_gid_ranges']:
            pop_gid_ranges[pop_name] = tuple(group['pop_gid_ranges'][pop_name][:])
        full_binned_t = group['full_binned_t'][:]
        buffered_binned_t = group['buffered_binned_t'][:]
        binned_t = group['binned_t'][:]
        subgroup = group['filter_bands']
        for filter in subgroup:
            filter_bands[filter] = subgroup[filter][:]
        full_rec_t = group['full_rec_t'][:]
        buffered_rec_t = group['buffered_rec_t'][:]
        rec_t = group['rec_t'][:]
        subgroup = group['connection_weights']
        for target_pop_name in subgroup:
            connection_weights_dict[target_pop_name] = dict()
            for source_pop_name in subgroup[target_pop_name]:
                connection_weights_dict[target_pop_name][source_pop_name] = \
                    subgroup[target_pop_name][source_pop_name][:]
        if 'tuning_peak_locs' in group and len(group['tuning_peak_locs']) > 0:
            subgroup = group['tuning_peak_locs']
            for pop_name in subgroup:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(subgroup[pop_name]['target_gids'], subgroup[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        subgroup = group['connectivity']
        for target_pop_name in subgroup:
            connectivity_dict[target_pop_name] = dict()
            for target_gid_key in subgroup[target_pop_name]:
                target_gid = int(target_gid_key)
                connectivity_dict[target_pop_name][target_gid] = dict()
                for source_pop_name in subgroup[target_pop_name][target_gid_key]:
                    connectivity_dict[target_pop_name][target_gid][source_pop_name] = \
                        subgroup[target_pop_name][target_gid_key][source_pop_name][:]
        subgroup = group['pop_syn_proportions']
        for target_pop_name in subgroup:
            pop_syn_proportions[target_pop_name] = dict()
            for syn_type in subgroup[target_pop_name]:
                pop_syn_proportions[target_pop_name][syn_type] = dict()
                source_pop_names = subgroup[target_pop_name][syn_type]['source_pop_names'][:].astype('str')
                for source_pop_name, syn_proportion in zip(source_pop_names,
                                                           subgroup[target_pop_name][syn_type]['syn_proportions'][:]):
                    pop_syn_proportions[target_pop_name][syn_type][source_pop_name] = syn_proportion
        subgroup = group['pop_cell_positions']
        for pop_name in subgroup:
            pop_cell_positions[pop_name] = dict()
            for gid, position in zip(subgroup[pop_name]['gids'][:], subgroup[pop_name]['positions'][:]):
                pop_cell_positions[pop_name][gid] = position
        group = get_h5py_group(f, [model_label, exported_data_key])
        subgroup = group['full_spike_times']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        subgroup = group['buffered_firing_rates']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                buffered_firing_rates_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        subgroup = group['buffered_firing_rates_from_binned_spike_count']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                buffered_firing_rates_from_binned_spike_count_dict[pop_name][int(gid_key)] = \
                    subgroup[pop_name][gid_key][:]
        subgroup = group['full_binned_spike_count']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                full_binned_spike_count_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]
        subgroup = group['subset_full_voltage_recs']
        for pop_name in subgroup:
            for gid_key in subgroup[pop_name]:
                subset_full_voltage_rec_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]

    binned_dt = binned_t[1] - binned_t[0]
    full_pop_mean_rate_from_binned_spike_count_dict = \
        get_pop_mean_rate_from_binned_spike_count(full_binned_spike_count_dict, dt=binned_dt)
    _ = get_pop_activity_stats(buffered_firing_rates_from_binned_spike_count_dict, input_t=buffered_binned_t,
                               valid_t=buffered_binned_t, threshold=active_rate_threshold, plot=plot)
    _ = get_pop_bandpass_filtered_signal_stats(full_pop_mean_rate_from_binned_spike_count_dict, filter_bands,
                                               input_t=full_binned_t, valid_t=binned_t, output_t=binned_t,
                                               plot=plot, verbose=verbose)
    if plot:
        plot_inferred_spike_rates(full_spike_times_dict, buffered_firing_rates_from_binned_spike_count_dict,
                                  input_t=buffered_binned_t, valid_t=buffered_binned_t,
                                  active_rate_threshold=active_rate_threshold)
        plot_voltage_traces(subset_full_voltage_rec_dict, full_rec_t, valid_t=rec_t,
                            spike_times_dict=full_spike_times_dict)
        plot_weight_matrix(connection_weights_dict, pop_gid_ranges=pop_gid_ranges, tuning_peak_locs=tuning_peak_locs)
        plot_firing_rate_heatmaps(buffered_firing_rates_from_binned_spike_count_dict, input_t=buffered_binned_t,
                                  valid_t=buffered_binned_t, tuning_peak_locs=tuning_peak_locs)
        plot_firing_rate_heatmaps(full_binned_spike_count_dict, input_t=full_binned_t,
                                  valid_t=buffered_binned_t, tuning_peak_locs=tuning_peak_locs)
        if connectivity_type == 'gaussian':
            plot_2D_connection_distance(pop_syn_proportions, pop_cell_positions, connectivity_dict)

    if return_context:
        context = Context()
        context.update(locals())
        return context


def decode_position_from_offline_replay(run_data_file_path, replay_data_file_path, run_model_key='0',
                                        replay_model_keys=None, window_dur = 40., step_dur = 20., verbose=False):
    """

    :param run_data_file_path: str (path)
    :param replay_data_file_path: str (path)
    :param run_model_key: str
    :param replay_model_keys: list of str
    :param window_dur: float (ms)
    :param step_dur: float (ms)
    :param verbose: bool
    :return: nested dict
    """
    import numpy.matlib
    run_context = analyze_simple_network_results_from_file(run_data_file_path, run_model_key, return_context=True,
                                                           plot=False)
    run_binned_t = run_context.binned_t
    run_buffered_binned_t = run_context.buffered_binned_t
    valid_indexes = np.where((run_buffered_binned_t >= run_binned_t[0]) &
                             (run_buffered_binned_t <= run_binned_t[-1]))[0]

    run_firing_rates_dict = defaultdict(dict)
    sorted_gid_dict = defaultdict(dict)
    for pop_name in run_context.buffered_firing_rates_dict:
        for gid_key in run_context.buffered_firing_rates_dict[pop_name]:
            run_firing_rates_dict[pop_name][int(gid_key)] = \
                run_context.buffered_firing_rates_dict[pop_name][gid_key][valid_indexes]
        if pop_name in run_context.tuning_peak_locs:
            this_target_gids = np.array(list(run_context.tuning_peak_locs[pop_name].keys()))
            this_peak_locs = np.array(list(run_context.tuning_peak_locs[pop_name].values()))
            indexes = np.argsort(this_peak_locs)
            sorted_gid_dict[pop_name] = this_target_gids[indexes]
        else:
            this_target_gids = [int(gid_key) for gid_key in run_context.buffered_firing_rates_dict[pop_name]]
            sorted_gid_dict[pop_name] = np.array(sorted(this_target_gids), dtype='int')

    with h5py.File(replay_data_file_path, 'r') as f:
        replay_input_t = f['shared_context']['full_binned_t'][:]
        replay_valid_t = f['shared_context']['buffered_binned_t'][:]
        binned_dt = replay_input_t[1] - replay_input_t[0]
        if replay_model_keys is None:
            replay_model_keys = []
            for model_key in f:
                if model_key != 'shared_context' and 'simple_network_exported_data' in f[model_key]:
                    replay_model_keys.append(model_key)
        else:
            for model_key in replay_model_keys:
                if model_key not in f:
                    raise RuntimeError('decode_position_from_offline_replay: model_key: %s not found in '
                                       'replay_data_file: %s' % (model_key, replay_data_file_path))

    valid_indexes = np.where((replay_input_t >= replay_valid_t[0]) & (replay_input_t <= replay_valid_t[-1]))[0]
    align_to_t = 0.

    half_window_bins = int(window_dur // binned_dt // 2)
    window_bins = int(2 * half_window_bins + 1)
    window_dur = window_bins * binned_dt

    step_bins = step_dur // binned_dt
    step_dur = step_bins * binned_dt
    half_step_dur = step_dur / 2.

    # if possible, include a bin centered on time zero.
    binned_t_center_indexes = []
    this_center_index = np.where(replay_valid_t >= align_to_t)[0]
    if len(this_center_index) > 0:
        this_center_index = this_center_index[0]
        if this_center_index < half_window_bins:
            this_center_index = half_window_bins
            binned_t_center_indexes.append(this_center_index)
        else:
            while this_center_index > half_window_bins:
                binned_t_center_indexes.append(this_center_index)
                this_center_index -= step_bins
            binned_t_center_indexes.reverse()
    else:
        this_center_index = half_window_bins
        binned_t_center_indexes.append(this_center_index)
    this_center_index = binned_t_center_indexes[-1] + step_bins
    while this_center_index < len(replay_valid_t) - half_window_bins:
        binned_t_center_indexes.append(this_center_index)
        this_center_index += step_bins

    binned_t_center_indexes = np.array(binned_t_center_indexes, dtype='int')
    decode_binned_t = replay_valid_t[binned_t_center_indexes]
    replay_x_mesh, replay_y_mesh = np.meshgrid(decode_binned_t - half_step_dur, run_binned_t)

    p_pos_dict = defaultdict(dict)
    binned_spike_count_matrix_dict = defaultdict(dict)
    for model_key in replay_model_keys:
        replay_context = analyze_simple_network_replay_results_from_file(replay_data_file_path, model_key,
                                                                         return_context=True, plot=False)
        full_binned_spike_count_dict = replay_context.full_binned_spike_count_dict

        for pop_name in full_binned_spike_count_dict:
            if len(full_binned_spike_count_dict[pop_name]) != len(run_firing_rates_dict[pop_name]):
                raise RuntimeError('decode_position_from_offline_replay: population: %s; mismatched number of cells to'
                                   ' decode')
            binned_spike_count = np.empty((len(full_binned_spike_count_dict[pop_name]), len(replay_valid_t)))
            run_firing_rates = np.empty((len(run_firing_rates_dict[pop_name]), len(run_binned_t)))
            for i, gid in enumerate(sorted_gid_dict[pop_name]):
                binned_spike_count[i, :] = full_binned_spike_count_dict[pop_name][gid][valid_indexes]
                run_firing_rates[i, :] = run_firing_rates_dict[pop_name][gid]

            p_pos = np.empty((len(run_binned_t), len(decode_binned_t)))
            p_pos.fill(np.nan)

            population_rate_discount = np.exp(-window_dur / 1000. * np.sum(run_firing_rates, axis=0, dtype='float128'))
            for p_pos_index, t_center_index in enumerate(binned_t_center_indexes):
                t_start_index = t_center_index - half_window_bins
                t_end_index = t_center_index + half_window_bins + 1
                local_spike_count_array = np.sum(binned_spike_count[:, t_start_index:t_end_index], axis=1,
                                                 dtype='float128')
                if len(np.where(local_spike_count_array > 0)[0]) > 1:
                    n = np.matlib.repmat(local_spike_count_array, len(run_binned_t), 1).T
                    this_p_pos = (run_firing_rates ** n).prod(axis=0) * population_rate_discount
                    this_p_sum = np.nansum(this_p_pos)
                    if np.isnan(this_p_sum):
                        p_pos[:, p_pos_index] = np.nan
                    elif this_p_sum > 0.:
                        p_pos[:, p_pos_index] = this_p_pos / this_p_sum
                    else:
                        p_pos[:, p_pos_index] = np.nan
            p_pos_dict[model_key][pop_name] = p_pos
            binned_spike_count_matrix_dict[model_key][pop_name] = binned_spike_count
        if verbose:
            print('decode_position_from_offline_replay: processed model: %s from replay_file_path: %s' %
                  (model_key, replay_data_file_path))
            sys.stdout.flush()

    return replay_valid_t, binned_spike_count_matrix_dict, replay_x_mesh, replay_y_mesh, decode_binned_t, p_pos_dict


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

    n = len(spktimes)
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

    rate = np.zeros((len(time),))
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