"""
Defines the FOGIDiagram class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.linalg as _spl
import collections as _collections
from ..baseobjs import Basis as _Basis
from ..models.fogistore import FirstOrderGaugeInvariantStore as _FOGIStore
from ..tools import matrixtools as _mt

import matplotlib.cm as _matplotlibcm
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap

try:
    import drawSvg as _draw
except ImportError:
    _draw = None
#_Hcmap = _matplotlibcm.get_cmap('Reds_r')
#_Scmap = _matplotlibcm.get_cmap('Blues_r')

_cdict = {'red': [[0.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0]],
          'green': [[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0]],
          'blue': [[0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0]]}
_Hcmap = _LinearSegmentedColormap('lightReds', segmentdata=_cdict, N=256)

_cdict = {'red': [[0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0]],
          'blue': [[0.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0]],
          'green': [[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0]]}
_Scmap = _LinearSegmentedColormap('lightBlues', segmentdata=_cdict, N=256)

## - create table/heatmap of relational & gate-local strengths - also filter by ham/sto
# - create table of all quantities to show structure


def _create_errgen_op(vec, list_of_mxs):
    return sum([c * mx for c, mx in zip(vec, list_of_mxs)])


def _is_dependent(infos_by_type):
    for typ, infos_by_actedon in infos_by_type.items():
        for acted_on, infos in infos_by_actedon.items():
            for info in infos:
                if info['dependent'] is False:
                    return False
    return True


def _dstr(d, joinstr="<br>"):  # dict-to-string formatting function
    if len(d) == 1: return "%.3g" % _np.real_if_close(next(iter(d.values())))
    return joinstr.join(["%s: %.3g" % (k, _np.real_if_close(v)) for k, v in d.items()])


def _fmt_tableval(val):
    if isinstance(val, dict):
        if len(val) == 1: return _fmt_tableval(next(iter(val.values())))
        return " <br> ".join(["%s: %s" % (k, _fmt_tableval(v)) for k, v in val.items()])
    if _np.iscomplex(val): val = _np.real_if_close(val)
    if _np.isreal(val) or _np.iscomplex(val):
        if abs(val) < 1e-6: val = 0.0
        if _np.isreal(val): return "%.3g" % val.real
        else: return "%.3g + %.3gj" % (val.real, val.imag)
    return str(val)


def _make_table(table_info, rowlbl, title):
    table_dict, table_rows, table_cols = table_info
    html = "<table><thead><tr><th colspan=%d>%s</th></tr>\n" % (len(table_cols) + 1, title)
    html += ("<tr><th>%s<th>" % rowlbl) + "</th><th>".join(table_cols) + "</th></tr></thead><tbody>\n"
    for row in table_rows:
        table_row_text = []
        for col in table_cols:
            val = table_dict[row][col]
            table_row_text.append(_fmt_tableval(val))
        html += "<tr><th>" + str(row) + "</th><td>" + "</td><td>".join(table_row_text) + "</td></tr>\n"
    return html + "</tbody></table>"


class FOGIDiagram(object):
    """
    A diagram of the first-order-gauge-invariant (FOGI) quantities of a model.

    This class encapsulates a way of visualizing a model's FOGI quantities.
    """

    def __init__(self, fogi_store, op_coefficients, model_dim, op_to_target_qubits=None, impact_mode='current'):
        # Note: fogi_store can one or multiple (a list of) stores
        self.impact_mode = impact_mode
        self.fogi_store = fogi_store
        self.fogi_comps = fogi_store.opcoeffs_to_fogi_components_array(op_coefficients)
        self.fogi_infos = fogi_store.create_binned_fogi_infos()

        #Construct op_to_target_qubits if needed
        if op_to_target_qubits is None:
            all_qubits = set()
            for op_label in fogi_store.primitive_op_labels:
                if op_label.sslbls is not None:
                    all_qubits.update(op_label.sslbls)
            all_qubits = tuple(sorted(all_qubits))
            op_to_target_qubits = {op_label: op_label.sslbls if (op_label.sslbls is not None) else all_qubits
                                   for op_label in self.fogi_store.primitive_op_labels}
        self.op_to_target_qubits = op_to_target_qubits

        #We need the gauge basis to construct actual gauge generators for computation of J-angle below.
        # these are un-normalized when we construct the gauge action in, e.g. first_order_ham_gauge_action_matrix(...)
        # and so they must be un-normalized (standard Pauli mxs) here.
        normalized_pauli_basis = _Basis.cast('pp', model_dim)
        scale = model_dim**(0.25)  # to change to standard pauli-product matrices
        self.gauge_basis_mxs = [mx * scale for mx in normalized_pauli_basis.elements[1:]]

        # op-sets (sets of operations) correspond to units/nodes on diagrams, so it's useful
        # to have dictionaries of all the summarized information about all the fogi quantities
        # living on a given set of operations.
        self.op_set_info = {}
        for op_set, op_fogi_infos_by_type in self.fogi_infos.items():

            total = {}
            flat_H_infos = [info for acted_on, infos in op_fogi_infos_by_type.get(('H',), {}).items() for info in infos]
            flat_S_infos = [info for acted_on, infos in op_fogi_infos_by_type.get(('S',), {}).items() for info in infos]
            if len(flat_H_infos) > 0: total['H'] = self._contrib(('H',), op_set, flat_H_infos)
            if len(flat_S_infos) > 0: total['S'] = self._contrib(('S',), op_set, flat_S_infos)
            total['mag'] = sum([self._extract_mag(contrib) for contrib in total.values()])

            self.op_set_info[op_set] = {
                'total': total,
                'hs_support_table': self._make_coherent_stochastic_by_support_table(op_set, op_fogi_infos_by_type),
                'individual_fogi_table': self._make_individual_fogi_table(op_set, op_fogi_infos_by_type),
                'abbrev_individual_fogi_table': self._make_abbrev_table(op_set, op_fogi_infos_by_type),
                'byweight': self._compute_by_weight_magnitudes(op_set, op_fogi_infos_by_type),
                'bytarget': self._compute_by_target_magnitudes(op_set, op_fogi_infos_by_type),
                'dependent': _is_dependent(op_fogi_infos_by_type),
            }
            assert(('H', 'S') not in op_fogi_infos_by_type)

    def _contrib(self, typ, op_set, infos_to_aggregate):
        def _sto_local_contrib(opt_set, infos):
            error_rate = 0
            for info in infos:
                fogi_vec = info['fogi_dir'] / _np.linalg.norm(info['fogi_dir'])**2
                vec_rate = sum(_np.abs(fogi_vec))  # L1 norm gives error rate of vector
                #sto rates should all have normalized fogi_dirs so rate = 1 == L1 norm of *vec* = *dir* / L2(*dir*)^2
                assert(_np.isclose(vec_rate, 1.0))  # if we've normalized correctly, this should be true
                error_rate += self.fogi_comps[info['fogi_index']] * vec_rate
            return {'error_rate': error_rate}

        def _sto_relational_contrib(opt_set, infos):
            error_rate = 0
            for info in infos:
                fogi_vec = info['fogi_dir'] / _np.linalg.norm(info['fogi_dir'])**2
                vec_rate = sum(fogi_vec)  # sum of elements gives error rate of vector
                # HERE - intrinsic relational rate addition is still a bit mysterious -- we should allow for
                # negative rates to balance positive ones to get an overall error rate, but maybe it's more
                # complicated than just adding teh elements up as we do above...
                error_rate += self.fogi_comps[info['fogi_index']] * vec_rate
            assert(abs(error_rate.imag) < 1e-6)
            return {'error_rate': error_rate.real}  # maybe negative rates are ok (?) -- we could take abs here.

        def _ham_local_contrib(op_set, infos):
            assert(len(op_set) == 1)
            op_label = tuple(op_set)[0]

            # Compute j-angle(error generated by sum_i(comp_i * e_i))  where
            # i = info index and e_i = (normalized) fogi direction of i-th fogi qty.
            if len(infos) == 0: return {op_label: 0.0, 'min_jangle': 0.0}
            op_indices_slc = self.fogi_store.op_errorgen_indices[op_set[0]]
            errgen_vec = _np.zeros((op_indices_slc.stop - op_indices_slc.start), complex)
            for info in infos:
                assert(set(op_set) == info['op_set'])
                comp = self.fogi_comps[info['fogi_index']]
                fogi_vec = info['fogi_dir'] / _np.linalg.norm(info['fogi_dir'])**2
                # fogi vec = "inverse" of fogi dir because all local fogi dirs are orthonormal but not necessarily
                # normalized - so dividing by the norm^2 here => dot(fogi_dir, fogi_vec) = 1.0 as desired.
                # (we treat fogi_dir as if it were a member of an orthogonal basis when adding contributions)
                errgen_vec += comp * fogi_vec[op_indices_slc]
            errgen_op = _create_errgen_op(errgen_vec, self.gauge_basis_mxs)  # NOTE: won't work for reduced models
            j_angle = _mt.jamiolkowski_angle(errgen_op)
            return {op_label: j_angle, 'min_jangle': j_angle}

        def _ham_relational_contrib(op_set, infos):
            if len(infos) == 0:
                ret = {'min_jangle': 0.0}
                ret.update({op_label: 0.0 for op_label in op_set})
                return ret

            # Compute j-angle(error generated by gauge-action-on-OP( sum_i(theta_i * gauge_vec_i)) )
            #  for each OP in op_set, where gauge_vec_i is the (normalized) gauge directions
            #  corresponding to the i-th fogi quantity (info).

            E = _np.column_stack([info['gauge_dir'] for info in infos])
            theta = _np.array([self.fogi_comps[info['fogi_index']] / info['r_factor'] for info in infos])
            gauge_dir = _np.dot(_np.linalg.pinv(E).T, theta)

            #OLD - doesn't deal with non-orthogonal gauge vecs properly,
            #      and I think forgets to renorm gauge_dir => gauge_vec
            #gauge_dir = None
            #for info in infos:
            #    assert(set(op_set) == info['op_set'])
            #    assert(info['gauge_dir'] is not None)
            #    comp = self.fogi_comps[info['fogi_index']]
            #    gauge_comp = comp / info['r_factor']  # "theta" -- component = r * theta
            #    gauge_vec = info['gauge_dir']
            #
            #    if gauge_dir is None:
            #        gauge_dir = gauge_comp * gauge_vec
            #    else:
            #        gauge_dir += gauge_comp * gauge_vec

            # get "impact", i.e., j-angle, for relational qtys

            # Compute the j-angle of the gauge op alone, as this is a potentially useful
            # way to add the theta angles of multiple quantities (Note, however, that this
            # is the j-angle of a gauge-space vector rather than an errorgen-space vector
            gauge_op = _create_errgen_op(gauge_dir, self.gauge_basis_mxs)  # gauge transform as a gauge-errorgen vec
            ret = {'go_jangle': _mt.jamiolkowski_angle(gauge_op)}
            if self.impact_mode == 'max':
                for op_label in op_set:
                    errgen_vec = _np.dot(self.fogi_store.gauge_action_for_op[op_label], gauge_dir)
                    errgen_op = _create_errgen_op(errgen_vec, self.gauge_basis_mxs)  # WON'T work for reduced models
                    ret[op_label] = _mt.jamiolkowski_angle(errgen_op)  # impact angle for op_label
            elif self.impact_mode == 'current':
                # use partition the current gauge's errorgen-vec, after projecting onto the given (relational) FOGI
                # directions, among the operations and compute their j-angles
                fogi_dirs = _np.column_stack([info['fogi_dir'] for info in infos])
                comps = _np.array([self.fogi_comps[info['fogi_index']] for info in infos])
                projected_e = _np.dot(_np.linalg.pinv(fogi_dirs).T, comps)
                for op_label in op_set:
                    errgen_vec = projected_e[self.fogi_store.op_errorgen_indices[op_label]]
                    errgen_op = _create_errgen_op(errgen_vec, self.gauge_basis_mxs)  # WON'T work for reduced models
                    ret[op_label] = _mt.jamiolkowski_angle(errgen_op)  # impact angle for op_label

            ret['min_jangle'] = min([v for k, v in ret.items() if k != 'go_jangle'])
            return ret

        if typ == ('H',):
            return _ham_local_contrib(op_set, infos_to_aggregate) if len(op_set) == 1 \
                else _ham_relational_contrib(op_set, infos_to_aggregate)
        elif typ == ('S',):
            return _sto_local_contrib(op_set, infos_to_aggregate) if len(op_set) == 1 \
                else _sto_relational_contrib(op_set, infos_to_aggregate)
        else:
            raise ValueError("Unknown error types set: %s" % str(typ))

    def _extract_mag(self, d):
        """ Extract the singe most important magnitude from a dictionary of contribution values
            such as those returned by _contrib(...) """
        if 'error_rate' in d: return d['error_rate']  # where we know error rate already (stochastic terms)
        else: return d['min_jangle']  # minimal j-angle (hamiltonian terms)

    def _make_coherent_stochastic_by_support_table(self, op_set, infos_by_type):
        table = {}
        table_rows = set()
        table_cols = set()
        for typ, infos_by_actedon in infos_by_type.items():
            if typ == ('H',): col = "Coherent"
            elif typ == ('S',): col = "Stochastic"
            else: col = "Mixed"

            table_cols.add(col)
            for acted_on, infos in infos_by_actedon.items():
                contrib = self._contrib(typ, op_set, infos)
                table_rows.add(acted_on)
                if acted_on not in table: table[acted_on] = {}
                table[acted_on][col] = contrib
        ordered_rows = tuple(sorted(table_rows, key=lambda t: (len(t),) + t))
        ordered_cols = tuple(sorted(table_cols))
        return table, ordered_rows, ordered_cols

    def _make_individual_fogi_table(self, op_set, infos_by_type):
        table = {}
        table_rows = []
        table_cols = ('Component', 'Contribution', 'Raw Label')
        for typ, infos_by_actedon in infos_by_type.items():
            for acted_on, infos in infos_by_actedon.items():
                for info in infos:
                    contrib = self._contrib(typ, op_set, [info])
                    table_rows.append(info['label'])
                    table[info['label']] = {'Component': self.fogi_comps[info['fogi_index']],
                                            'Raw Label': info['label_raw'],
                                            'Contribution': contrib}
        return table, tuple(table_rows), table_cols

    def _make_abbrev_table(self, op_set, infos_by_type):
        abbrev_label_coeff_list = []
        for typ, infos_by_actedon in infos_by_type.items():
            for acted_on, infos in infos_by_actedon.items():
                for info in infos:
                    contrib = self._contrib(typ, op_set, [info])
                    theta = _np.real_if_close(self.fogi_comps[info['fogi_index']] / info['r_factor']) \
                        if (info['r_factor'] != 0.0) else None
                    abbrev_label_coeff_list.append((info['label_abbrev'],
                                                    self.fogi_comps[info['fogi_index']], theta,
                                                    contrib))  # computed "aggregate" contributions but for single qty
        return abbrev_label_coeff_list

    def _compute_by_weight_magnitudes(self, op_set, infos_by_type):
        infos_by_typ_weight = {}
        for typ, infos_by_actedon in infos_by_type.items():
            if typ not in infos_by_typ_weight:
                infos_by_typ_weight[typ] = _collections.defaultdict(list)
            for acted_on, infos in infos_by_actedon.items():
                if all([set(acted_on).issubset(self.op_to_target_qubits[op]) for op in op_set]):
                    infos_by_typ_weight[typ][0].extend(infos)  # local == weight 0
                else:
                    infos_by_typ_weight[typ][len(acted_on)].extend(infos)  # higher weight

        mags_by_weight = {}  # "weight" of term is an int.  0 = local
        weights = sorted(set([x for d in infos_by_typ_weight.values() for x in d.keys()]))

        for weight in weights:
            mag = 0
            items_abbrev = []
            for typ, infos_by_weight in infos_by_typ_weight.items():
                if weight in infos_by_weight:
                    total_dict = self._contrib(typ, op_set, infos_by_weight[weight])
                    mag += self._extract_mag(total_dict)
                    items_abbrev.extend([(info['label_abbrev'], self.fogi_comps[info['fogi_index']],
                                          self._contrib(typ, op_set, [info])) for info in infos_by_weight[weight]])

            mags_by_weight[weight] = {'total': mag,
                                      'items': items_abbrev}  # = list of (abbrev-item-lbl, coeff, contrib)
        return mags_by_weight

    def _compute_by_target_magnitudes(self, op_set, infos_by_type):
        infos_by_typ_target = {}
        for typ, infos_by_actedon in infos_by_type.items():
            if typ not in infos_by_typ_target:
                infos_by_typ_target[typ] = _collections.defaultdict(list)
            for acted_on, infos in infos_by_actedon.items():
                infos_by_typ_target[typ][acted_on].extend(infos)

        targets = sorted(set([x for d in infos_by_typ_target.values() for x in d.keys()]))

        mags_by_target = {}
        for target in targets:
            mag = 0
            items_abbrev = []
            for typ, infos_by_target in infos_by_typ_target.items():
                if target in infos_by_target:
                    total_dict = self._contrib(typ, op_set, infos_by_target[target])
                    mag += self._extract_mag(total_dict)
                    items_abbrev.extend([(info['label_abbrev'], self.fogi_comps[info['fogi_index']],
                                          self._contrib(typ, op_set, [info])) for info in infos_by_target[target]])

            mags_by_target[target] = {'total': mag,
                                      'items': items_abbrev}  # = list of (abbrev-item-lbl, coeff, contrib)
        return mags_by_target


class FOGIGraphDiagram(FOGIDiagram):
    def __init__(self, fogi_stores, op_coefficients, model_dim, op_to_target_qubits=None,
                 physics=True, numerical_labels=False, edge_threshold=1e-6, color_mode="separate",
                 node_fontsize=20, edgenode_fontsize=14, edge_fontsize=12, impact_mode='current'):
        # color_mode can also be "mix" and "mix_wborder"
        super().__init__(fogi_stores, op_coefficients, model_dim, op_to_target_qubits, impact_mode)
        self.physics = physics
        self.numerical_labels = numerical_labels
        self.edge_threshold = edge_threshold
        self.color_mode = color_mode
        self.node_fontsize = node_fontsize
        self.edgenode_fontsize = edgenode_fontsize
        self.edge_fontsize = edge_fontsize
        self.MIN_POWER = 1.5  # 10^-MIN_POWER is the largest value end of the spectrum
        self.MAX_POWER = 3.5  # 10^-MAX_POWER is the smallest value end of the spectrum

    def _normalize(self, v):
        return -_np.log10(max(v, 10**(-self.MAX_POWER)) * 10**self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)

    #def _normalize(v):
    #    v = min(max(v, 10**(-MAX_POWER)), 10**(-MIN_POWER))
    #    return 1.0 - v / (10**(-MIN_POWER) - 10**(-MAX_POWER))

    def _node_HScolor(self, Hvalue, Svalue):
        r, g, b, a = _Hcmap(self._normalize(Hvalue))
        r2, g2, b2, a2 = _Scmap(self._normalize(Svalue))
        r = (r + r2) / 2; g = (g + g2) / 2; b = (b + b2) / 2
        return "rgb(%d,%d,%d)" % (int(r * 255.9), int(g * 255.9), int(b * 255.9))

    def _node_Hcolor(self, value):
        r, g, b, a = _Hcmap(self._normalize(value))
        return "rgb(%d,%d,%d)" % (int(r * 255.9), int(g * 255.9), int(b * 255.9))

    def _node_Scolor(self, value):
        r, g, b, a = _Scmap(self._normalize(value))
        return "rgb(%d,%d,%d)" % (int(r * 255.9), int(g * 255.9), int(b * 255.9))
        #if value < 1e-5: return "rgb(133,173,133)"  # unsaturaged green
        #if value < 3e-5: return "rgb(0,230,0)"  # green
        #if value < 1e-4: return "rgb(115,230,0)"  # light green
        #if value < 3e-4: return "rgb(172,230,0)"  # lime green
        #if value < 1e-3: return "rgb(250,250,0)"  # yellow
        #if value < 3e-3: return "rgb(255,204,0)"  # yellow-orange
        #if value < 1e-2: return "rgb(255,153,0)"  # orange
        #if value < 3e-2: return "rgb(255,140,26)"  # dark orange
        #if value < 1e-1: return "rgb(255,102,0)"  # orange-red
        #if value < 3e-1: return "rgb(255,102,51)"  # red-orange
        #return "rgb(255,0,0)"  # red

        #if value < 1e-3: return "green"
        #if value < 1e-2: return "yellow"
        #if value < 1e-1: return "orange"
        #return "red"

    def _color_to_textcolor(self, color_str):
        textcolors = ['black', 'white']
        r, g, b = tuple(map(int, color_str[len('rgb('):-1].split(',')))
        lightness = sum([r, g, b]) / (3 * 256)
        return textcolors[int(lightness < 0.5)]

    def _get_node_colors(self, coh, sto, total, compact=False, edge_node=False):
        color_mode = self.color_mode
        labels = []
        make_labels = bool(self.numerical_labels is True
                           or (isinstance(self.numerical_labels, (tuple, list)) and 'node' in self.numerical_labels))

        if compact and 'H' in total and 'S' in total:  # eliminate 'H' or 'S' if they're zero
            total = total.copy()
            if coh <= self.edge_threshold: del total['H']
            elif sto <= self.edge_threshold: del total['S']

        if 'H' in total and 'S' in total:
            title = "Coherent: %.3g<br>Stochastic: %.3g" % (coh, sto)
            if color_mode == "mix" or color_mode == "fixed":
                back_color = border_color = self._node_HScolor(coh, sto)
            elif color_mode == "mix_wborder":
                back_color = self._node_HScolor(coh, sto); border_color = "rgb(100, 100, 100)"
            elif color_mode == "separate":
                back_color = self._node_Scolor(sto); border_color = self._node_Hcolor(coh)
            if make_labels:
                labels.extend(["H: %.1f%%" % (100 * coh), "S: %.1f%%" % (100 * sto)])
            edge_color = self._node_HScolor(coh, sto)  # edges coming from this node are always of the "mix" color

        elif 'H' in total:
            title = "%.3g" % coh
            back_color = border_color = edge_color = self._node_Hcolor(coh)
            if color_mode == "mix_wborder": border_color = "rgb(100, 100, 100)"
            if make_labels: labels.append("%.1f%%" % (100 * coh))
        elif 'S' in total:
            title = "%.3g" % sto
            back_color = border_color = edge_color = self._node_Scolor(sto)
            if color_mode == "mix_wborder": border_color = "rgb(100, 100, 100)"
            if make_labels: labels.append("%.1f%%" % (100 * sto))
        else:
            raise ValueError("Invalid types in total dict: %s" % str(total.keys()))
        tcolor = self._color_to_textcolor(back_color)

        if self.color_mode == "fixed":  # special case that overrides automatic colors
            back_color = border_color = edge_color = '#FF9090'
            tcolor = 'black'

        return back_color, border_color, tcolor, edge_color, labels, title

    def render(self, filename, initial_theta=0, custom_polar_positions=None, custom_polar_ranges=None):
        drawing = self._create_drawing()

        op_set_info = self.op_set_info
        op_labels = self.fogi_store.primitive_op_labels  # take just from first store

        #Group ops based on target qubits
        target_qubit_groups = tuple(sorted(set(self.op_to_target_qubits.values())))
        groupids = {op_label: target_qubit_groups.index(self.op_to_target_qubits[op_label])
                    for op_label in op_labels}
        groups = {group_id: [] for group_id in range(len(target_qubit_groups))}
        for op_label in op_labels:
            groups[groupids[op_label]].append(op_label)

        # Position ops around a circle with more space between groups)
        nPositions = len(op_labels) + len(groups)  # includes gaps between groups
        r0 = 270

        polar_positions = {}; group_theta_ranges = {}
        if custom_polar_positions is None:
            theta = initial_theta; dtheta = 2 * _np.pi / nPositions
            for group_id, ops_in_group in groups.items():
                theta_begin = theta
                for op_label in ops_in_group:
                    polar_positions[op_label] = (r0, theta)
                    theta += dtheta
                group_theta_ranges[group_id] = (theta_begin - dtheta / 3, theta - dtheta + dtheta / 3)
                theta += dtheta
        else:
            for group_id, ops_in_group in groups.items():
                thetas = [custom_polar_positions[op_label] for op_label in ops_in_group]
                for op_label in ops_in_group:
                    polar_positions[op_label] = (r0, custom_polar_positions[op_label])
                if custom_polar_ranges:
                    for op_label in ops_in_group:
                        if op_label in custom_polar_ranges:
                            group_theta_ranges[group_id] = custom_polar_ranges[op_label]
                            break
                    else:
                        raise ValueError("Missing polar range for group with %s" % str(ops_in_group))
                else:
                    group_theta_ranges[group_id] = (min(thetas) - _np.pi / 10, max(thetas) + _np.pi / 10)

        # Background wedges
        for group_id, (theta_begin, theta_end) in group_theta_ranges.items():
            target_qubits = target_qubit_groups[group_id]
            txt = ("Qubit %d" % target_qubits[0]) if len(target_qubits) == 1 \
                else ("Qubits " + ", ".join(map(str, target_qubits)))
            #print(txt, "at theta = ", txt_theta)  # DEBUG
            self._draw_wedge(drawing, r0, theta_begin, theta_end, txt)

        #process local quantities
        node_locations = {}
        val_max = max([(abs(info['total'].get('H', {}).get('min_jangle', 0))
                        + info['total'].get('S', {}).get('error_rate', 0))
                       for info in op_set_info.values()])
        #val_threshold = _normalize(val_max) / 3.0

        for op_set, info in op_set_info.items():
            if len(op_set) != 1: continue
            # create a gate-node in the graph
            r, theta = polar_positions[op_set[0]]

            total = info['total']
            coh = total['H']['min_jangle'] if ('H' in total) else 0.0
            sto = total['S']['error_rate'] if ('S' in total) else 0.0
            #print("VAL = ",(abs(coh) + sto),"Threshold = ",val_threshold)
            self._draw_node(drawing, r, theta, coh, sto, op_set[0], total, val_max, groupids[op_label], info)
            node_locations[op_set[0]] = r * _np.cos(theta), r * _np.sin(theta)

        #process relational quantities
        relational_distances = []
        for op_set, info in op_set_info.items():
            if len(op_set) == 1: continue
            max_dist = 0
            for i in range(len(op_set)):
                x1, y1 = node_locations[op_set[i]]
                for j in range(i + 1, len(op_set)):
                    x2, y2 = node_locations[op_set[j]]
                    max_dist = max(max_dist, (x1 - x2)**2 + (y1 - y2)**2)
            relational_distances.append((op_set, max_dist))
        relational_opsets_by_distance = sorted(relational_distances, key=lambda x: x[1], reverse=True)

        #place the longest nPositions linking nodes in the center; place the rest on the periphery
        for i, (op_set, _) in enumerate(relational_opsets_by_distance):
            info = op_set_info[op_set]
            total = info['total']
            mag = total['mag']
            if mag < self.edge_threshold: continue

            # create a relational node in the graph
            if True or i < nPositions:  # place node in the middle (just average coords
                x = _np.mean([node_locations[op_label][0] for op_label in op_set])
                y = _np.mean([node_locations[op_label][1] for op_label in op_set])
            else:  # place node along periphery
                r = r0 * 1.1  # push outward from ring of operations
                theta = _np.arctan2(_np.sum([node_locations[op_label][1] for op_label in op_set]),
                                    _np.sum([node_locations[op_label][0] for op_label in op_set]))
                x, y = r * _np.cos(theta), r * _np.sin(theta)

            coh = total['H']['go_jangle'] if ('H' in total) else None
            sto = total['S']['error_rate'] if ('S' in total) else None
            edge_color = self._draw_edge_node(drawing, x, y, coh, sto, total, val_max, op_set, info)

            edge_labels = {}
            make_labels = bool(self.numerical_labels is True
                               or (isinstance(self.numerical_labels, (tuple, list))
                                   and 'edge' in self.numerical_labels))
            if make_labels and abs(coh) > self.edge_threshold:
                #Only Hamiltonian edge labels so far (TODO)
                edge_labels = {op_label: total['H'][op_label] for op_label in op_set}

            for op_label in op_set:
                node_x, node_y = node_locations[op_label]
                self._draw_edge(drawing, x, y, node_x, node_y, edge_color, mag, val_max,
                                edge_labels.get(op_label, None), op_label, info['dependent'])

        return self._render_drawing(drawing, filename)


class FOGISvgGraphDiagram(FOGIGraphDiagram):

    def __init__(self, fogi_stores, op_coefficients, model_dim, op_to_target_qubits=None,
                 physics=True, numerical_labels=False, edge_threshold=1e-6, color_mode="separate",
                 node_fontsize=20, edgenode_fontsize=14, edge_fontsize=12, impact_mode='current'):
        super().__init__(fogi_stores, op_coefficients, model_dim, op_to_target_qubits, physics, numerical_labels,
                         edge_threshold, color_mode, node_fontsize, edgenode_fontsize, edge_fontsize, impact_mode)

        self.tweaks = {
            # 'node_width_ab': (20, 80),  # (a,b) => a*x + b
            # 'node_height_ab': (100, 10),
            'node_size_ab': (10000, 0),  # scales *area*
            'node_rounded_rect_radius': 10,
            'op_lbl_roffset': 120,
            'node_fontsize_ab': (1.3, 0.5),
            'edgenode_size_ab': (5000, 0),  # scales *area*
            'edgenode_fontsize_ab': (1.3, 0.5),
            'edge_width_ab': (40, 3.0),
            'edge_label_offset': 0.6,
            'wedge_fontsize': 15,
            'wedge_padding_factor': 0.2,
            'wedge_width': 100,
            'wedge_control_mag': 150.0,  # adjusts control points for banana curves
            'wedge_label_offset': 80,
            'svg_size': (800, 800)
        }

    def _create_drawing(self):
        SVGDrawing = _collections.namedtuple('SVGDrawing', ['nodes', 'edges', 'edgelines'])
        return SVGDrawing([], [], [])

    def _render_drawing(self, drawing, filename):
        sizex, sizey = self.tweaks['svg_size']
        d = _draw.Drawing(sizex, sizey, origin='center', displayInline=False)
        for x in drawing.edges: d.append(x)
        for x in drawing.edgelines: d.append(x)
        for x in drawing.nodes: d.append(x)
        if filename: d.saveSvg(filename)
        return d

    #def _draw_node_simple(self, drawing, r, theta, coh, sto, op_label, total, val_max):
    #    nodes = drawing.nodes
    #    back_color, border_color, tcolor, _, labels, _ = self._get_node_colors(coh, sto, total)
    #    x, y = r * _np.cos(theta), r * _np.sin(theta)
    #    scale = (coh + sto) / val_max
    #    node_width = 20 + 40 * scale
    #    node_height = 20 + 40 * scale
    #    nodes.append(_draw.Rectangle(x - node_width / 2, y - node_height / 2, node_width, node_height, rx=3,
    #                                 fill=back_color, stroke=border_color, stroke_width=2))
    #    nodes.append(_draw.Text(labels, self.node_fontsize * (0.5 + scale), x, y, fill=tcolor,
    #                            text_anchor="middle", valign='middle', font_family='Times'))

    def _draw_node(self, drawing, r, theta, coh, sto, op_label, total, val_max, groupid, info):
        nodes = drawing.nodes
        back_color, border_color, tcolor, _, labels, _ = self._get_node_colors(coh, sto, total)

        if self.color_mode == "fixed":
            back_color = 'blue'
            border_color = 'red'
            tcolor = 'white'

        #wa, wb = self.tweaks['node_width_ab']
        #ha, hb = self.tweaks['node_height_ab']
        sa, sb = self.tweaks['node_size_ab']
        x, y = r * _np.cos(theta), r * _np.sin(theta)
        scale = (coh + sto) / val_max

        # `scale` scales the *length* of nodes
        #node_width = wb + wa * scale
        #sto_rect_height = hb + ha * sto / val_max
        #ham_rect_height = hb + ha * coh / val_max
        #node_height = sto_rect_height + ham_rect_height

        # `scale` scales the *area*
        area = sb + sa * scale
        node_width = node_height = _np.sqrt(area)
        sto_rect_height = (area * sto / (coh + sto)) / node_width
        ham_rect_height = (area * coh / (coh + sto)) / node_width
        # sum = (A1 + A2) / node_width = area * 1 / node_width = node_height (good!)

        r_op_lbl = r + self.tweaks['op_lbl_roffset']
        rx = self.tweaks['node_rounded_rect_radius']

        nodes.append(_draw.Rectangle(x - node_width / 2, y - node_height / 2, node_width, sto_rect_height, rx=rx,
                                     fill=back_color, stroke='none'))
        nodes.append(_draw.Rectangle(x - node_width / 2, y - node_height / 2 + sto_rect_height / 2, node_width,
                                     sto_rect_height / 2, fill=back_color, stroke='none'))  # non-rounded rect
        nodes.append(_draw.Rectangle(x - node_width / 2, y - node_height / 2 + sto_rect_height, node_width,
                                     ham_rect_height, rx=rx, fill=border_color, stroke='none'))
        nodes.append(_draw.Rectangle(x - node_width / 2, y - node_height / 2 + sto_rect_height, node_width,
                                     ham_rect_height / 2, fill=border_color, stroke='none'))  # non-rounded rect
        nodes.append(_draw.Rectangle(x - node_width / 2, y - node_height / 2, node_width, node_height, rx=rx,
                                     fill='none', stroke='black', stroke_width=1))

        if len(labels) > 0:
            ham_lbl, sto_lbl = labels
            fa, fb = self.tweaks['node_fontsize_ab']
            nodes.append(_draw.Text(ham_lbl, self.node_fontsize * (fb + fa * coh / val_max), x,
                                    y - node_height / 2 + sto_rect_height + ham_rect_height / 2, fill=tcolor,
                                    text_anchor="middle", valign='middle', font_family='Times'))
            nodes.append(_draw.Text(sto_lbl, self.node_fontsize * (fb + fa * sto / val_max), x,
                                    y - node_height / 2 + sto_rect_height / 2, fill=tcolor,
                                    text_anchor="middle", valign='middle', font_family='Times'))
            nodes.append(_draw.Text(str(op_label), 2 * self.node_fontsize,
                                    r_op_lbl * _np.cos(theta), r_op_lbl * _np.sin(theta),
                                    fill='black', text_anchor="middle", valign='middle', font_family='Times'))

    def _draw_edge_node(self, drawing, x, y, coh, sto, total, val_max, op_set, info):
        nodes = drawing.nodes
        back_color, border_color, tcolor, edge_color, labels, _ = self._get_node_colors(coh, sto, total, compact=True)
        if coh is None: coh = 0
        if sto is None: sto = 0
        scale = (coh + sto) / val_max
        sa, sb = self.tweaks['edgenode_size_ab']

        #node_radius = sb + sa * scale  # `scale` scales radius
        node_radius = _np.sqrt((sb + sa * scale) / _np.pi)  # `scale` scales area

        nodes.append(_draw.Circle(x, y, node_radius,
                                  fill=back_color, stroke_width=1, stroke="black"))  # border_color
        if len(labels) > 0:
            fa, fb = self.tweaks['edgenode_fontsize_ab']
            nodes.append(_draw.Text(labels, self.edgenode_fontsize * (fb + fa * scale), x, y, fill=tcolor,
                                    text_anchor="middle", valign='middle', font_family='Times'))
        return edge_color

    def _draw_edge(self, drawing, x1, y1, x2, y2, edge_color, mag, val_max, edge_label, op_label, is_dependent):
        edges = drawing.edges
        edgelines = drawing.edgelines
        wa, wb = self.tweaks['edge_width_ab']
        edge_width = wb + wa * (mag / val_max)
        main_edge = _draw.Path(stroke=edge_color, stroke_width=edge_width, fill='none')
        main_edge.M(x1, y1).L(x2, y2)  # moveTo, lineTo
        edges.append(main_edge)

        is_dependent = False  # HACK for paper - disable dashed lines
        line_edge = _draw.Path(stroke="black", stroke_width=min(edge_width / 2, 0.5), fill='none',
                               stroke_dasharray="4" if is_dependent else 'none')
        line_edge.M(x1, y1).L(x2, y2)  # moveTo, lineTo
        edgelines.append(line_edge)
        #edges.append(_draw.Lines(x, y, node_x, node_y, close=False,
        #                        fill='none', stroke='gray', stroke_width=1))

        if edge_label is not None:
            txt_edge = _draw.Path(stroke='none', stroke_width=0, fill='none')
            if x1 <= x2: txt_edge.M(x1, y1).L(x2, y2)  # moveTo, lineTo
            else: txt_edge.M(x2, y2).L(x1, y1)  # Swap ordering so text gets drawn rightside-up
            prefix = "max " if self.impact_mode == 'max' else ""
            edgelines.append(_draw.Text("%s%.1f%%" % (prefix, 100 * edge_label),
                                        self.edge_fontsize, path=txt_edge, center=True, fill='black',
                                        font_family='Times', lineOffset=self.tweaks['edge_label_offset']))

    def _draw_wedge(self, drawing, r, theta_begin, theta_end, txt):

        to_degrees = 360.0 / (2 * _np.pi)
        txt_fontsize = self.tweaks['wedge_fontsize']

        rpadding = r * self.tweaks['wedge_padding_factor']
        #txt_theta = (theta_begin + theta_end) / 2
        #x = (r0 + rpadding + 20) * _np.cos(txt_theta)
        #y = (r0 + rpadding + 20) * _np.sin(txt_theta)  # text location
        #txt_angle = (txt_theta + _np.pi) if (0 <= txt_theta <= _np.pi) else txt_theta

        edges = drawing.edges
        p = _draw.Path(fill='#CCCCCC', stroke='none')

        typ = "bananna"  # "pie_slice"
        if typ == "pie_slice":
            p.arc(0, 0, r + rpadding, theta_begin * to_degrees, theta_end * to_degrees, cw=False)
            p.arc(0, 0, 0, theta_end * to_degrees, theta_begin * to_degrees, cw=True, includeL=True)
            p.Z()  # closepath
            edges.append(p)

        elif typ == "bananna":
            bananna_width = self.tweaks['wedge_width']
            r_high = r + rpadding
            r_low = r_high - bananna_width
            #r_mid = (r_high + r_low) / 2.0
            #dtheta = _np.pi / 10
            ctrl_mag = self.tweaks['wedge_control_mag']

            p.arc(0, 0, r_high, theta_begin * to_degrees, theta_end * to_degrees, cw=False)
            bx, by = r_high * _np.cos(theta_end), r_high * _np.sin(theta_end)  # begin point
            cx1, cy1 = bx - ctrl_mag * _np.sin(theta_end), by + ctrl_mag * _np.cos(theta_end)  # control pt for begin
            ex, ey = r_low * _np.cos(theta_end), r_low * _np.sin(theta_end)  # end point
            cx2, cy2 = ex - ctrl_mag * _np.sin(theta_end), ey + ctrl_mag * _np.cos(theta_end)  # control pt for begin
            p.C(cx1, cy1, cx2, cy2, ex, ey)

            p.arc(0, 0, r_low, theta_end * to_degrees, theta_begin * to_degrees, cw=True, includeL=True)
            bx, by = r_low * _np.cos(theta_begin), r_low * _np.sin(theta_begin)  # begin point
            cx1, cy1 = bx + ctrl_mag * _np.sin(theta_begin), by - ctrl_mag * _np.cos(theta_begin)  # ctrl pt for begin
            ex, ey = r_high * _np.cos(theta_begin), r_high * _np.sin(theta_begin)  # end point
            cx2, cy2 = ex + ctrl_mag * _np.sin(theta_begin), ey - ctrl_mag * _np.cos(theta_begin)  # ctrl pt for end
            p.C(cx1, cy1, cx2, cy2, ex, ey)

            p.Z()  # closepath
            edges.append(p)

        #Draw text along arc
        roffset = self.tweaks['wedge_label_offset']
        dtheta = _np.pi / 6  # make path extra large so we ensure there's enough path to fit entire label
        theta_begin -= dtheta
        theta_end += dtheta
        if _np.cos(theta_begin) < _np.cos(theta_end):
            arc = _draw.Arc(0, 0, r + roffset, theta_begin * to_degrees, theta_end * to_degrees, stroke='none',
                            fill='none', cw=False)
        else:
            arc = _draw.Arc(0, 0, r + roffset, theta_end * to_degrees, theta_begin * to_degrees, stroke='none',
                            fill='none', cw=True)
        edges.append(arc)
        edges.append(_draw.Text(txt, txt_fontsize, path=arc, fill="#CCCCCC",
                                text_anchor="middle", valign='middle', font_family='Times'))


class FOGIVisNetworkGraphDiagram(FOGIGraphDiagram):

    def _create_drawing(self):
        VisDrawing = _collections.namedtuple('VisNetworkDrawing',
                                             ['beforeDrawingCalls', 'node_js_lines', 'edge_js_lines',
                                              'table_html', 'long_table_html', 'node_ids',
                                              'next_node_id'])
        return VisDrawing([], [], [], {}, {}, {}, [0])

    def _render_drawing(self, drawing, filename):
        all_table_html = ""
        for node_id, html in drawing.table_html.items():
            all_table_html += ("<div class='infotable dataTable' id='%d'>\n" % node_id) + html + "</div>\n"

        all_long_table_html = ""
        for node_id, html in drawing.long_table_html.items():
            all_long_table_html += ("<div class='infotable dataTable' id='long%d'>\n" % node_id) + html + "</div>\n"

        springlength = 1.0  # r0 * dtheta / 3.0  # heuristic
        s = _template.format(**{'node_js': ",\n".join(drawing.node_js_lines),
                                'edge_js': ",\n".join(drawing.edge_js_lines),
                                'table_html': all_table_html,
                                'long_table_html': all_long_table_html,
                                'physics': 'true' if self.physics else "false",
                                'springlength': springlength,
                                'beforeDrawingCalls': "".join(drawing.beforeDrawingCalls),
                                'node_fontsize': self.node_fontsize,
                                'edge_fontsize': self.edge_fontsize})
        if filename:
            with open(filename, 'w') as f:
                f.write(s)

    def _draw_node(self, drawing, r, theta, coh, sto, op_label, total, val_max, groupid, info):
        back_color, border_color, tcolor, _, labels, title = self._get_node_colors(coh, sto, total)

        label = "\\n".join([str(op_label)] + labels)
        next_node_id = drawing.next_node_id[0]
        drawing.node_js_lines.append(('{id: %d, label: "%s", group: %d, title: "%s", x: %d, y: %d,'
                                      'font: {color: "%s"}, color: {background: "%s", border: "%s"}, fixed: %s}') %
                                     (next_node_id, label, groupid,
                                      title, int(r * _np.cos(theta)), int(r * _np.sin(theta)), tcolor, back_color,
                                      border_color, 'true' if self.physics else 'false'))
        drawing.table_html[next_node_id] = _make_table(info['hs_support_table'], "Qubits",
                                                       "Local errors on %s" % str(op_label))
        drawing.long_table_html[next_node_id] = _make_table(info['individual_fogi_table'],
                                                            "Label", "FOGI quantities for %s" % str(op_label))
        drawing.node_ids[op_label] = next_node_id
        drawing.next_node_id[0] += 1

    def _draw_edge_node(self, drawing, x, y, coh, sto, total, val_max, op_set, info):
        back_color, border_color, tcolor, edge_color, labels, title = self._get_node_colors(coh, sto, total,
                                                                                            compact=True)
        if coh is None: coh = 0
        if sto is None: sto = 0
        x = int(x)  # I think vis.network needs integer positions
        y = int(y)

        next_node_id = drawing.next_node_id[0]
        label = "\\n".join(labels)
        drawing.node_js_lines.append(('{ id: %d, label: "%s", group: "%s", title: "%s", x: %d, y: %d,'
                                      'color: {background: "%s", border: "%s"}, font: {size: %d, '
                                      'strokeWidth: 5, strokeColor: "white"} }') %
                                     (next_node_id, label, "relational", title, x, y, back_color, border_color,
                                      self.edgenode_fontsize))
        drawing.table_html[next_node_id] = _make_table(info['hs_support_table'], "Qubits",
                                                       "Relational errors between " + ", ".join(map(str, op_set)))
        drawing.long_table_html[next_node_id] = _make_table(info['individual_fogi_table'], "Label",
                                                            "FOGI quantities for " + ", ".join(map(str, op_set)))
        drawing.next_node_id[0] += 1
        return edge_color

    def _draw_edge(self, drawing, x1, y1, x2, y2, edge_color, mag, val_max, edge_label, op_label, is_dependent):
        if edge_label is not None:
            prefix = "max " if self.impact_mode == 'max' else ""
            label_str = ', label: "%s%.1f%%"' % (prefix, 100 * edge_label)
        else:
            label_str = ""

        edge_node_id = drawing.next_node_id[0] - 1  # Note: assumes that the last added node was this edge's node
        drawing.edge_js_lines.append('{ from: %d, to: %d, value: %.4f, color: {color: "%s"}, dashes: %s}' % (
            edge_node_id, drawing.node_ids[op_label], mag, edge_color, 'true' if is_dependent else 'false'))
        drawing.edge_js_lines.append('{ from: %d, to: %d, dashes: %s %s }' % (
            edge_node_id, drawing.node_ids[op_label], 'true' if is_dependent else 'false', label_str))

    def _draw_wedge(self, drawing, r, theta_begin, theta_end, txt):

        rpadding = 50
        txt_theta = (theta_begin + theta_end) / 2
        x = (r + rpadding + 20) * _np.cos(txt_theta)
        y = (r + rpadding + 20) * _np.sin(txt_theta)  # text location
        txt_angle = (txt_theta + _np.pi) if (0 <= txt_theta <= _np.pi) else txt_theta

        drawing.beforeDrawingCalls.append(('ctx.beginPath();\n'
                                           'ctx.arc(0, 0, %f, %f, %f, false);\n'
                                           'ctx.lineTo(0, 0);\n'
                                           'ctx.closePath();\n'
                                           'ctx.fill();\n'
                                           'ctx.save();\n'
                                           'ctx.translate(%f, %f);\n'
                                           'ctx.rotate(Math.PI / 2 + %f);\n'
                                           'ctx.fillText("%s", 0, 0);\n'
                                           'ctx.restore();\n') % (
                                               r + rpadding, theta_begin, theta_end, x, y, txt_angle, txt))


class FOGIDetailTable(FOGIDiagram):
    def __init__(self, fogi_stores, op_coefficients, model_dim, op_to_target_qubits=None,
                 mode='individual_terms', impact_mode='current'):
        super().__init__(fogi_stores, op_coefficients, model_dim, op_to_target_qubits, impact_mode)
        assert(mode in ('individual_terms', 'by_support'))
        self.mode = mode

    def render(self, filename):
        from .table import ReportTable as _ReportTable

        op_set_info = self.op_set_info
        op_labels = list(self.fogi_store.primitive_op_labels)

        op_label_lookup = {lbl: i for i, lbl in enumerate(op_labels)}
        cell_tabulars = {}

        def _dstr(d, op_set):  # dict-to-string formatting function
            d = {k: (v if abs(v) > 1e-6 else 0.0) for k, v in d.items()}  # copy d w/snap to 0.0
            if 'error_rate' in d:
                return "%.3g" % d['error_rate'] + " & " * (len(op_set) - 1)  # Stochastic
            else:
                return " & ".join(["%.3g" % d[op] for op in op_set])  # Hamiltonian

        additional = 0
        for op_set, info in op_set_info.items():
            if self.mode == 'individual_terms':
                rows = info['abbrev_individual_fogi_table']

                if len(rows) > 0:
                    rep_contrib = rows[0][3]  # rows[0] == lbl, component, theta, contrib
                    header = " & \\emph{comp}"
                    if 'error_rate' in rep_contrib:  # A stochastic quantity
                        header += " & \\emph{error rate} " + " & " * (len(op_set) - 1) + " \\\\ "
                    else:
                        if len(op_set) > 1:  # relational Hamiltonian quantities have a "theta"
                            header += " & \\emph{$\\theta$}"
                        header += " & " + " & ".join(map(str, op_set)) + " \\\\ "
                else:
                    header = ""

                ncols = 2 + len(op_set) + (1 if (len(op_set) > 1) else 0)  # if stmt adds extra col for theta value
                tabular = "\\begin{tabular}{@{}%s@{}}" % ('c' * ncols) + header \
                    + " \\\\ ".join(["%s & %.3g %s & %s" % (lbl, comp if abs(comp) > 1e-6 else 0.0,
                                                            ("& %.3g" % theta) if (theta is not None) else "",
                                                            _dstr(contrib, op_set))
                                     for lbl, comp, theta, contrib in info['abbrev_individual_fogi_table']]) \
                    + r"\end{tabular}"

                #Note: don't need to display go_jangle contrib quantities because these are just the theta values:
                #for lbl, _, theta, contrib in info['abbrev_individual_fogi_table']:
                #    if 'go_jangle' in contrib and not _np.isclose(abs(theta), contrib['go_jangle']):
                #        print("NOT CLOSE: ",lbl, theta, contrib['go_jangle'])
                assert(all([_np.isclose(abs(theta), contrib['go_jangle'], rtol=1e-3)
                            for _, _, theta, contrib in info['abbrev_individual_fogi_table']
                            if 'go_jangle' in contrib]))

            else:  # by-support
                raise NotImplementedError()

            if len(op_set) > 2:
                additional += 1
            elif len(op_set) == 2:
                i, j = op_label_lookup[op_set[0]], op_label_lookup[op_set[1]]
                if i > j: i, j = j, i  # ensure i <= j (upper triangle of table)
                cell_tabulars[(i, j)] = tabular
            else:
                i = op_label_lookup[op_set[0]]
                cell_tabulars[(i, i)] = tabular

        table = _ReportTable(['Gate'] + op_labels, [None] * (len(op_labels) + 1))
        #test_table_text = r"\begin{tabular}{@{}cc@{}}XZ & 0.5 \\ IZ & 1.0\end{tabular}"
        for i, op_label in enumerate(op_labels):
            table.add_row([str(op_label)] + [cell_tabulars.get((i, j), "")
                                             for j in range(len(op_labels))])  # ("%.2g" % data[i, j])
        table.finish()
        table_dict = table.render('latex')
        d = {'toLatex': table_dict['latex']}

        from .merge_helpers import merge_latex_template as _merge_latex_template
        _merge_latex_template(d, "standalone.tex", filename, {})
        print("Wrote latex file: %s" % filename)
        return table


class FOGIMultiscaleGridDiagram(FOGIDiagram):
    def __init__(self, fogi_stores, op_coefficients, model_dim, op_to_target_qubits=None, impact_mode='current'):
        super().__init__(fogi_stores, op_coefficients, model_dim, op_to_target_qubits, impact_mode)

    def render(self, detail_level=0, figsize=5, outfile=None, spacing=0.05, nudge=0.125,
               cell_fontsize=10, axes_fontsize=10, qty_key='bytarget'):
        op_set_info = self.op_set_info
        op_labels = list(self.fogi_store.primitive_op_labels)
        nOps = len(op_labels)

        op_label_lookup = {lbl: i for i, lbl in enumerate(op_labels)}
        totals = _np.ones((len(op_labels), len(op_labels)), 'd') * _np.nan
        by_qty_totals = {}
        by_qty_items = {}

        additional = 0
        for op_set, info in op_set_info.items():

            if len(op_set) == 2:
                i, j = op_label_lookup[op_set[0]], op_label_lookup[op_set[1]]
                if i > j: i, j = j, i
            else:
                i = j = op_label_lookup[op_set[0]]

            # Total value/contribution for entire "box"
            val = info['total']['mag']

            if len(op_set) > 2:
                additional += val
                continue

            assert(_np.isnan(totals[i, j]))
            totals[i, j] = val

            for qty, qty_dict in info[qty_key].items():
                #if weight > max_weight: continue  # with warning/accumulation?
                if qty not in by_qty_totals:
                    by_qty_totals[qty] = _np.ones((nOps, nOps), 'd') * _np.nan
                    by_qty_items[qty] = {(i, j): [] for i in range(nOps) for j in range(i, nOps)}
                assert(_np.isnan(by_qty_totals[qty][i, j]))
                by_qty_totals[qty][i, j] = qty_dict['total']
                by_qty_items[qty][i, j] = qty_dict['items']

        if qty_key == "bytarget":
            all_qtys = sorted(list(by_qty_totals.keys()), key=lambda k: (len(k),) + k)  # sort primarily by # targets
        else:
            all_qtys = sorted(list(by_qty_totals.keys()))
        nQtys = len(all_qtys)

        total_items = _np.zeros((nOps, nOps), 'i')
        for i in range(nOps):
            for j in range(i, nOps):
                total_items[i, j] = sum([len(by_qty_items[qty][i, j]) for qty in all_qtys])
                if total_items[i, j] == 0: totals[i, j] = _np.NaN

        box_size_mode = "condensed"  # or "inflated"
        if detail_level == 2:
            #compute the number of rows needed for each row in the heatmap
            max_items_by_qty = {qty: max([len(by_qty_items[qty][i, j]) for i in range(nOps) for j in range(i, nOps)])
                                for qty in all_qtys}
            max_total_items = _np.max(total_items)

            if box_size_mode == "inflated":
                box_size = sum(max_items_by_qty.values())
            else:
                box_size = max_total_items

        else:
            max_items_by_qty = None  # should be unused
            box_size = 1.0

        spacing *= box_size  # make spacing relative to box size
        row_y = _np.arange(len(op_labels) - 1, -1, -1) * (box_size + spacing) + spacing
        col_x = _np.arange(len(op_labels)) * (box_size + spacing) + spacing

        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        #min_color, max_color = 0, _np.nanmax(data)
        fig, axes = plt.subplots(1, 1, figsize=(figsize, figsize))
        axes.set_title("Errors on gates and between gate pairs (missing %.3g)" % (additional))
        axis_labels = [str(lbl) for lbl in op_labels]
        axes.set_xlim(0, len(op_labels) * (box_size + spacing) + spacing)
        axes.set_ylim(0, len(op_labels) * (box_size + spacing) + spacing)

        #im = axes.imshow(plot_data, cmap="Reds")
        #im.set_clim(min_color, max_color)

        # We want to show all ticks...
        axes.set_xticks(col_x + box_size / 2)
        axes.set_yticks(row_y + box_size / 2)
        # ... and label them with the respective list entries
        axes.set_xticklabels(axis_labels, fontsize=axes_fontsize)
        axes.set_yticklabels(axis_labels, fontsize=axes_fontsize)

        # Rotate the tick labels and set their alignment.
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Normalize the threshold to the images color range.
        textcolors = ['black', 'white']
        cmap = _matplotlibcm.get_cmap('Greys')
        cmap_by_qty = [_matplotlibcm.get_cmap('Reds'),
                       _matplotlibcm.get_cmap('Greens'),
                       _matplotlibcm.get_cmap('Blues')]  # more?
        val_max = _np.nanmax(totals)
        val_threshold = val_max / 2.
        nudge *= spacing  # how much to nudge borders into spacing area

        def create_box(ax, x, y, side, val, val_by_qty, items_by_qty):
            if detail_level == 0:
                if not _np.isnan(val):
                    val_color = cmap(val / val_max)[0:3]  # drop alpha
                    rect = patches.Rectangle((x, y), side, side, linewidth=0, edgecolor='k', facecolor=val_color)
                    ax.add_patch(rect)
                    ax.text(x + side / 2, y + side / 2, "%.1f%%" % (val * 100), ha="center", va="center",
                            color=textcolors[int(val > val_threshold)], fontsize=cell_fontsize)
                border = patches.Rectangle((x - nudge, y - nudge), side + 2 * nudge, side + 2 * nudge,
                                           linewidth=4, edgecolor='k', fill=False)
                ax.add_patch(border)

            elif detail_level == 1:
                section_size = side / nQtys
                y2 = y + section_size * (nQtys - 1)
                for iqty, qty in enumerate(all_qtys):
                    if qty_key == "byweight":
                        if qty == 0: nm = "Local"
                        elif qty == 1: nm = "Spectator"
                        else: nm = "Weight-%d" % qty
                    elif qty_key == "bytarget":
                        nm = ", ".join([str(qlbl) for qlbl in qty])
                    else:
                        nm = str(qty)

                    v = val_by_qty[qty]
                    if not _np.isnan(v):
                        cm = cmap_by_qty[iqty]
                        val_color = cm(v / val_max)[0:3]  # drop alpha
                        rect = patches.Rectangle((x, y2), side, section_size, linewidth=1, edgecolor='k',
                                                 facecolor=val_color)
                        ax.add_patch(rect)
                        ax.text(x + side / 2, y2 + section_size / 2, "%s = %.1f%%" % (nm, v * 100),
                                ha="center", va="center", color=textcolors[int(v > val_threshold)],
                                fontsize=cell_fontsize)
                    y2 -= section_size
                border = patches.Rectangle((x - nudge, y - nudge), side + 2 * nudge, side + 2 * nudge,
                                           linewidth=4, edgecolor='k', fill=False)
                ax.add_patch(border)

            elif detail_level == 2:
                y2 = y + side  # marks *top* of section here
                for iqty, qty in enumerate(all_qtys):
                    items = items_by_qty[qty]
                    if box_size_mode == "inflated":
                        section_size = max_items_by_qty[qty]
                    else:
                        section_size = len(items)
                    cm = cmap_by_qty[iqty]

                    y3 = y2
                    for lbl, coeff, contrib in items:
                        contrib_mag = self._extract_mag(contrib)
                        if abs(contrib_mag) < 1e-6: contrib_mag = 0.0

                        y3 -= 1
                        val_color = cm(contrib_mag / val_max)[0:3]  # drop alpha
                        rect = patches.Rectangle((x, y3), side, 1.0, linewidth=0, edgecolor='k',
                                                 facecolor=val_color)
                        ax.add_patch(rect)
                        ax.text(x + side / 2, y3 + 1 / 2, "%s = %.1f%%" % (lbl, contrib_mag * 100),
                                ha="center", va="center", color=textcolors[int(contrib_mag > val_threshold)],
                                fontsize=cell_fontsize)
                    y2 -= section_size
                    border = patches.Rectangle((x, y2), side, section_size, linewidth=1, edgecolor='k', fill=False)
                    ax.add_patch(border)
                border = patches.Rectangle((x - nudge, y - nudge), side + 2 * nudge, side + 2 * nudge,
                                           linewidth=4, edgecolor='k', fill=False)
                ax.add_patch(border)

            else:
                raise ValueError("Invalid `detail_level`!")

        # Loop over data dimensions and create text annotations.
        for i in range(len(axis_labels)):
            for j in range(len(axis_labels)):
                if i > j: continue
                create_box(axes, col_x[j], row_y[i], box_size, totals[i, j],
                           {qty: by_qty_totals[qty][i, j] for qty in all_qtys},
                           {qty: by_qty_items[qty][i, j] for qty in all_qtys})

        fig.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()
        return fig


class FOGIStackedBarDiagram(FOGIDiagram):
    def __init__(self, fogi_stores, op_coefficients, model_dim, op_to_target_qubits=None, impact_mode='current'):
        super().__init__(fogi_stores, op_coefficients, model_dim, op_to_target_qubits, impact_mode)

    def render_svg(self, outfile=None, cell_fontsize=10, axes_fontsize=10, qty_key='bytarget'):
        #d = _draw.Drawing(600, 600, origin='center', displayInline=False)

        #barplot_infos = {}  # want to index this dict by [op_label][err_type][intrinsic_or_relative][support][Species]
        # and have result be a list of ?
        for op_set, op_fogi_infos_by_type in self.fogi_infos.items():
            pass


_template = """<html>
<head>
    <!-- <link rel="stylesheet" href="pygsti_dataviz.css"> -->
    <!-- <script type="text/javascript" src="vis-network.js"></script> -->
    <!-- <script type="text/javascript" src="jquery-3.2.1.min.js"></script> -->
    <script type="text/javascript" src="vis-network.js"></script>
    <script type="text/javascript" src="jquery-3.2.1.min.js"></script>
    <!-- <script type="text/javascript"
                 src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script> -->
    <!-- <script src="https://code.jquery.com/jquery-3.5.1.min.js"
                 integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=" crossorigin="anonymous"></script> -->

    <style type="text/css">
      body {{
          //color: #d3d3d3;
          font: 12pt arial;
          //background-color: #222222;
      }}

      #mynetwork {{
          width: 800px;
          height: 800px;
          border: 1px solid #444444;
          //background-color: #222222;
      }}

      .invistable td {{
          vertical-align: top;
      }}

      .infotable {{
          display: none;
      }}

      .infotable table {{
          border-collapse: collapse;
      }}

      .dataTable td, .dataTable th {{
         border: 2px solid #333;
         padding: 5px;
      }}

      .dataTable {{
          font-family: "Computer Modern Serif","times new roman",serif;
          border-spacing: 0px;
          border-collapse: collapse;
          width: auto;
          margin: auto;
          margin-top: 5px;
          max-width: 100%;
          overflow: auto; /*hidden; - this hides content sometimes - needed?  */
      }}

      .dataTable tbody tr td {{
          background-color: #ffffff;
          color: #262c31;
          font-family: "Computer Modern Serif","times new roman",serif;
          font-size: 10pt;
          text-align: center;
          vertical-align: middle;
      }}

      .dataTable thead tr th,
      .dataTable tfoot tr td {{
          background-color: #333;
          color: #ffffff;
          font-size: 12px;
          font-weight: bold;
          text-align: center;
      }}
    </style>
</head>
<body>
<H2>FOGI Model Visualizer (experimental)</H2>
<table class="invistable">
<tr><td>
<div id="mynetwork"></div>
<!-- <pre id="eventSpan">TEST</pre> -->
</td><td>
<div id="tables">
{table_html}
</div>
<div id="longtables">
{long_table_html}
</div>
</td></tr></table>

<script type="text/javascript">
var nodes = null;
var edges = null;
var network = null;

function draw() {{
  // create people.
  // value corresponds with the age of the person
  nodes = new vis.DataSet([
{node_js}
  ]);

  // create connections between people
  // value corresponds with the amount of contact between two people
  edges = new vis.DataSet([
{edge_js}
  ]);

  // Instantiate our network object.
  var container = document.getElementById("mynetwork");
  var data = {{
    nodes: nodes,
    edges: edges,
  }};
  var options = {{
      nodes: {{
          shape: "box",
          size: 30,
          font: {{
              multi: "html",
              size: {node_fontsize},
          }},
          color: {{highlight: {{ background: "white", border: "black" }},
                   hover: {{ background: "white", border: "black" }} }},
          borderWidth: 3,
      }},
      edges: {{
          smooth: false,
          length: {springlength},
          width: 2,
          color: {{color: "gray", highlight: "black"}},
          scaling: {{min: 6, max: 14, label: {{enabled: false}} }}, // was  4 -> 10
          font: {{
              size: {edge_fontsize},
              color: "rgb(100, 100, 100)",
          }},
      }},
      groups: {{
          "relational": {{
              shape: "dot",
              size: 6,
              color: "gray",
          }}
      }},
      interaction: {{ hover: true, dragNodes: true, zoomView: false, dragView: true, multiselect: true  }},
      manipulation: {{
          enabled: false,
      }},
      physics: {{
          enabled: {physics},
          solver: "repulsion",
          repulsion: {{nodeDistance: 100, springLength: {springlength}, springConstant: 0.05}},
      }},
  }};
  network = new vis.Network(container, data, options);

  //network.moveTo({{
  //  position: {{x: 0, y: 0}},
  //  offset: {{x: -800/2, y: -800/2}},
  //  scale: 1,
  //}});

  network.on("click", function (params) {{
    $(".infotable").hide()
    for(var i = 0; i < params.nodes.length; i++) {{
        $("#" + params.nodes[i]).show()
    }}
    //document.getElementById("eventSpan").innerHTML =
    //  "<h2>Click event: node</h2>" + params.nodes;
    //console.log(
    //  "click event, getNodeAt returns: " + this.getNodeAt(params.pointer.DOM)
    //);
  }});

  network.on("doubleClick", function (params) {{
    $(".infotable").hide()
    for(var i = 0; i < params.nodes.length; i++) {{
        $("#" + params.nodes[i]).show()
        $("#long" + params.nodes[i]).show()
    }}
  }});

  network.on("beforeDrawing", function (ctx) {{
    //ctx.strokeStyle = "#A6D5F7";
    ctx.fillStyle = "#CCCCCC";
    ctx.font = "20px Georgia";
    ctx.textAlign = "center"
    ctx.textBaseline = "middle";
    {beforeDrawingCalls}
  }});

}}

window.addEventListener("load", () => {{
  draw();
}});
</script>
</body>
</html>
"""
