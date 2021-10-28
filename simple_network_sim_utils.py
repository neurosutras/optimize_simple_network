from nested.utils import *
from neuron import h
from scipy.signal import hann
from collections import namedtuple, defaultdict
from simple_network_analysis_utils import *


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
                          spikes_seed=None, tuning_duration=None):
        """

        :param input_types: dict
        :param input_mean_rates: dict
        :param input_min_rates: dict
        :param input_max_rates: dict
        :param input_norm_tuning_widths: dict
        :param tuning_peak_locs: dict
        :param track_wrap_around: bool
        :param spikes_seed: list of int: random seed for reproducible input spike trains
        :param tuning_duration: float
        """
        if spikes_seed is None:
            raise RuntimeError('SimpleNetwork.set_input_pattern: missing spikes_seed required to generate reproducible'
                               ' spike trains')
        if self.equilibrate > 0.:
            equilibrate_len = int(self.equilibrate/self.dt)
            equilibrate_rate_array = hann(int(self.equilibrate * 2. / self.dt))[:equilibrate_len]
        for pop_name in (pop_name for pop_name in input_types if pop_name in self.cells):
            if input_types[pop_name] == 'constant':
                if input_mean_rates is None or pop_name not in input_mean_rates:
                    raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_mean_rates required to specify '
                                       '%s input population: %s' % (input_types[pop_name], pop_name))
                this_mean_rate = input_mean_rates[pop_name]
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
                local_np_random = np.random.default_rng(seed=spikes_seed + [gid])
                this_spike_train = \
                    get_inhom_poisson_spike_times_by_thinning(self.input_pop_firing_rates[pop_name][gid],
                                                              self.input_pop_t[pop_name], dt=self.dt,
                                                              generator=local_np_random)
                cell = self.cells[pop_name][gid]
                cell.load_vecstim(this_spike_train)

    def set_offline_input_pattern(self, input_offline_min_rates=None, input_offline_max_rates=None,
                                  input_offline_fraction_active=None, tuning_peak_locs=None, track_wrap_around=False,
                                  stim_epochs=None, selection_seed=None, spikes_seed=None,
                                  tuning_duration=None, debug=False):
        """

        :param input_offline_min_rates: dict
        :param input_offline_max_rates: dict
        :param input_offline_fraction_active: dict
        :param tuning_peak_locs: dict
        :param track_wrap_around: bool
        :param stim_epochs: dict of list of tuple of float: {pop_name: [(epoch_start, epoch_duration, epoch_type)]
        :param selection_seed: list of int: random seed for reproducible input cell selection
        :param spikes_seed: list of int: random seed for reproducible input spike trains
        :param tuning_duration: float
        :param debug: bool
        """
        if spikes_seed is None:
            raise RuntimeError('SimpleNetwork.set_offline_input_pattern: missing spikes_seed required to generate '
                               'reproducible spike trains')
        if selection_seed is None:
            raise RuntimeError('SimpleNetwork.set_offline_input_pattern: missing selection_seed required to select '
                               'reproducible ensembles of active inputs')
        if stim_epochs is None or len(stim_epochs) == 0:
            return
        else:
            stim_defined = False
            for pop_name in stim_epochs:
                if len(stim_epochs[pop_name]) > 0:
                    stim_defined = True
            if not stim_defined:
                return

        for pop_name in (pop_name for pop_name in stim_epochs if pop_name in self.cells):
            if input_offline_min_rates is None or pop_name not in input_offline_min_rates:
                raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_offline_min_rates required to '
                                   'specify input population: %s' % pop_name)
            if input_offline_max_rates is None or pop_name not in input_offline_max_rates:
                raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_offline_max_rates required to '
                                   'specify input population: %s' % pop_name)
            if input_offline_fraction_active is None or pop_name not in input_offline_fraction_active:
                raise RuntimeError('SimpleNetwork.set_input_pattern: missing input_offline_fraction_active '
                                   'required to specify input population: %s' % pop_name)
            this_min_rate = input_offline_min_rates[pop_name]
            this_max_rate = input_offline_max_rates[pop_name]
            this_fraction_active = input_offline_fraction_active[pop_name]

            stim_t = []
            stim_rate = []
            last_epoch_end = 0.
            for epoch_start, epoch_duration, epoch_type in stim_epochs[pop_name]:
                if epoch_start != last_epoch_end:
                    raise RuntimeError('set_offline_input_pattern: input population: %s; invalid start time: %.2f for '
                                       'stim epoch: %s' % (pop_name, epoch_start, epoch_type))
                epoch_end = min(epoch_start + epoch_duration, self.tstop)
                last_epoch_end = epoch_end
                if epoch_type == 'onset':
                    epoch_len = int(epoch_duration / self.dt)
                    epoch_t = np.arange(epoch_start, epoch_end, self.dt)
                    epoch_rate = \
                        (this_max_rate - this_min_rate) * hann(int(epoch_duration * 2. / self.dt))[:epoch_len] + \
                        this_min_rate
                elif epoch_type == 'offset':
                    epoch_len = int(epoch_duration / self.dt)
                    epoch_t = np.arange(epoch_start, epoch_end, self.dt)
                    epoch_rate = \
                        (this_max_rate - this_min_rate) * hann(int(epoch_duration * 2. / self.dt))[:epoch_len][::-1] + \
                        this_min_rate
                elif epoch_type == 'min':
                    epoch_t = [epoch_start, epoch_end]
                    epoch_rate = [this_min_rate, this_min_rate]
                elif epoch_type == 'max':
                    epoch_t = [epoch_start, epoch_end]
                    epoch_rate = [this_max_rate, this_max_rate]
                stim_t.append(epoch_t)
                stim_rate.append(epoch_rate)
            stim_t = np.concatenate(stim_t)
            stim_rate = np.concatenate(stim_rate)
            self.input_pop_t[pop_name] = stim_t
            for gid in self.cells[pop_name]:
                local_np_random = np.random.default_rng(seed=selection_seed + [gid])
                if local_np_random.uniform(0., 1.) <= this_fraction_active:
                    self.input_pop_firing_rates[pop_name][gid] = stim_rate
                else:
                    self.input_pop_firing_rates[pop_name][gid] = \
                        np.ones_like(self.input_pop_t[pop_name]) * this_min_rate

        for pop_name in (pop_name for pop_name in stim_epochs if pop_name in self.cells):
            for gid in self.cells[pop_name]:
                local_np_random = np.random.default_rng(seed=spikes_seed + [gid])
                this_spike_train = \
                    get_inhom_poisson_spike_times_by_thinning(self.input_pop_firing_rates[pop_name][gid],
                                                              self.input_pop_t[pop_name], dt=self.dt,
                                                              generator=local_np_random)
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

    def connect_cells(self, connectivity_type='uniform', connection_seed=None, **kwargs):
        """

        :param connectivity_type: str
        :param connection_seed: list of int: random seed for reproducible connections
        """
        if connection_seed is None:
            raise RuntimeError('SimpleNetwork.connect_cells: missing connection_seed required to generate reproducible'
                               ' connections')
        rank = int(self.pc.id())
        for target_pop_name in self.pop_syn_proportions:
            total_syn_count = self.pop_syn_counts[target_pop_name]
            for target_gid in self.cells[target_pop_name]:
                local_np_random = np.random.default_rng(seed=connection_seed + [target_gid])
                target_cell = self.cells[target_pop_name][target_gid]
                for syn_type in self.pop_syn_proportions[target_pop_name]:
                    for source_pop_name in self.pop_syn_proportions[target_pop_name][syn_type]:
                        p_syn_count = self.pop_syn_proportions[target_pop_name][syn_type][source_pop_name]
                        this_syn_count = local_np_random.binomial(total_syn_count, p_syn_count)
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

                        this_source_gids = local_np_random.choice(potential_source_gids, size=this_syn_count,
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
                                  connection_weight_distribution_types=None, weights_seed=None):
        """

        :param default_weight_distribution_type: str
        :param connection_weight_distribution_types: nested dict: {target_pop_name: {source_pop_name: str}}
        :param weights_seed: list of int: random seed for reproducible connection weights
        """
        if weights_seed is None:
            raise RuntimeError('SimpleNetwork.set_input_pattern: missing weights_seed required to assign reproducible'
                               ' synaptic weights')
        rank = int(self.pc.id())
        for target_pop_name in self.ncdict:
            for target_gid in self.ncdict[target_pop_name]:
                local_np_random = np.random.default_rng(seed=weights_seed + [target_gid])
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
                                    this_weight = mu * local_np_random.normal(1., norm_sigma)
                            elif this_weight_distribution_type == 'lognormal':
                                # enforce weights to be less than 5-fold greater than mean
                                this_weight = 5. * mu
                                while this_weight >= 5. * mu:
                                    this_weight = mu * local_np_random.lognormal(0., norm_sigma)
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
            if 'skew' in structured_weight_params[target_pop_name]:
                from scipy.stats import skewnorm
                this_sigma = this_tuning_width / 6.
                this_skew = structured_weight_params[target_pop_name]['skew']
                skewnorm_scale = structured_weight_params[target_pop_name]['skew_width'] * this_sigma
                skewnorm_x = np.linspace(skewnorm.ppf(0.01, this_skew, scale=skewnorm_scale),
                                         skewnorm.ppf(0.99, this_skew, scale=skewnorm_scale), 10000)
                skewnorm_pdf = skewnorm.pdf(skewnorm_x, this_skew, scale=skewnorm_scale)
                skewnorm_argmax = np.argmax(skewnorm_pdf)
                skewnorm_max = skewnorm_pdf[skewnorm_argmax]
                skewnorm_loc_shift = skewnorm_x[skewnorm_argmax]
                this_tuning_f = \
                    lambda delta_loc: this_peak_delta_weight / skewnorm_max * \
                                      skewnorm.pdf(delta_loc, this_skew, -skewnorm_loc_shift, skewnorm_scale)
            else:
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
                            this_delta_loc = tuning_peak_locs[source_pop_name][source_gid] - this_target_loc
                            if wrap_around:
                                if this_delta_loc > tuning_duration / 2.:
                                    this_delta_loc -= tuning_duration
                                elif this_delta_loc < - tuning_duration / 2.:
                                    this_delta_loc += tuning_duration
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
    def __init__(self, pop_name=None, gid=None, cell_type='RS', ):
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
