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
from ..objects import Basis as _Basis
import matplotlib.cm as _matplotlibcm
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
#_cmap = _matplotlibcm.get_cmap('Reds')

_cdict = {'red': [[0.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0]],
          'green': [[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0]],
          'blue': [[0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0]]}
_cmap = _LinearSegmentedColormap('lightReds', segmentdata=_cdict, N=256)

## - create table/heatmap of relational & gate-local strengths - also filter by ham/sto
# - create table of all quantities to show structure


class FOGIDiagram(object):
    """
    A diagram of the first-order-gauge-invariant (FOGI) quantities of a model.

    This class encapsulates a way of visualizing a model's FOGI quantities.
    """

    def __init__(self, model, error_type="both", op_to_target_qubits=None, debug=False):
        assert(error_type in ('both', 'hamiltonian', 'stochastic'))
        self.error_type = error_type
        self.fogi_info = model.fogi_info.copy()  # shallow copy
        self.fogi_coeffs = model.fogi_errorgen_coefficients_array(normalized_elem_gens=False)
        self.bins = self._create_binned_data()
        self.debug = debug

        if op_to_target_qubits is None:
            all_qubits = set()
            for op_label in self.fogi_info['primitive_op_labels']:
                if op_label.sslbls is not None:
                    all_qubits.update(op_label.sslbls)
            all_qubits = tuple(sorted(all_qubits))
            op_to_target_qubits = {op_label: op_label.sslbls if (op_label.sslbls is not None) else all_qubits
                                   for op_label in self.fogi_info['primitive_op_labels']}
        self.op_to_target_qubits = op_to_target_qubits

        normalized_pauli_basis = _Basis.cast('pp', model.dim)
        scale = model.dim**(0.25)  # to change to standard pauli-product matrices
        gauge_basis_mxs = [mx * scale for mx in normalized_pauli_basis.elements[1:]]

        def _spectral_radius(x):
            if hasattr(x, 'ndim') and x.ndim == 2:  # then interpret as a numpy array and take norm
                evals = _np.sort(_np.linalg.eigvals(x))
                return abs(evals[-1] - evals[0])
            else:
                return x

        def _jamiolkowski_angle(Hmx):
            d = Hmx.shape[0]
            I = _np.identity(d)
            errmap = _np.kron(I, _spl.expm(1j * Hmx))
            psi = _np.zeros(d**2)  # will be a maximally entangled state
            for i in range(d):
                x = _np.zeros(d); x[i] = 1.0
                xx = _np.kron(x, x)
                psi += xx / _np.sqrt(d)
            assert(_np.isclose(_np.dot(psi, psi), 1.0))
            cos_theta = abs(_np.dot(psi.conj(), _np.dot(errmap, psi)))
            return _np.arccos(cos_theta)
            #cos_squared_theta = entanglement_infidelity(expm(1j * Hmx), identity)
            #return _np.arccos(_np.sqrt(cos_squared_theta))

        def _create_errgen_op(vec, list_of_mxs):
            return sum([c * mx for c, mx in zip(vec, list_of_mxs)])

        def _total_contrib(typ, op_set, flat_infos):
            if typ == ('S',):
                return _sto_total_contrib(flat_infos)
            elif typ == ('H',):
                if len(op_set) == 1:
                    return _ham_total_local_contrib(op_set, flat_infos)
                else:
                    return _ham_total_relational_contrib(op_set, flat_infos)
            else:
                raise ValueError("Invalid type: " + str(typ))

        def _sto_total_contrib(flat_infos):
            return sum([abs(info['coeff']) for info in flat_infos])

        def _ham_total_local_contrib(op_set, flat_infos):
            assert(len(op_set) == 1)
            if len(flat_infos) == 0: return {'errgen_angle': 0.0}
            op_indices_slc = self.fogi_info['ham_op_errgen_indices'][op_set[0]]
            errgen_vec = _np.zeros((op_indices_slc.stop - op_indices_slc.start), complex)
            for info in flat_infos:
                assert(set(op_set) == info['ops'])
                coeff = info['coeff']
                inv_fogi_vec = info['fogi_vec'] / _np.linalg.norm(info['fogi_vec'])**2
                # "inverse" fogi vector because all local fogi vecs are orthonormal but not necessarily
                # normalized - so dividing by the norm^2 here => dot(inv_fogi_vec, fogi_vec) = 1.0
                errgen_vec += coeff * inv_fogi_vec[op_indices_slc]
            Hmx = _create_errgen_op(errgen_vec, gauge_basis_mxs)  # NOTE: won't work for reduced models
            angle = _jamiolkowski_angle(Hmx)
            return {'errgen_angle': angle}

        def _ham_total_relational_contrib(op_set, flat_infos):
            if len(flat_infos) == 0:
                if self.debug:
                    ret = {'go_angle': 0.0}
                    ret.update({op_label: 0.0 for op_label in op_set})
                    return ret
                else:
                    return {'min_impact': 0.0}

            gauge_vec = None
            for info in flat_infos:
                assert(set(op_set) == info['ops'])
                assert(info['gauge_dir'] is not None)
                if gauge_vec is None:
                    gauge_vec = info['coeff'] * info['gauge_dir']
                else:
                    gauge_vec += info['coeff'] * info['gauge_dir']

            # get "impact" for relational qtys
            Hmx = _create_errgen_op(gauge_vec, gauge_basis_mxs)
            ret = {'go_angle': _jamiolkowski_angle(Hmx)}
            for op_label in op_set:
                errgen_vec = _np.dot(self.fogi_info['ham_gauge_action_mxs'][op_label], gauge_vec)
                Hmx = _create_errgen_op(errgen_vec, gauge_basis_mxs)  # NOTE: won't work for reduced models
                ret[op_label] = _jamiolkowski_angle(Hmx)  # impact angle for op_label
            if self.debug: return ret
            else: return {'min_impact': min([v for k, v in ret.items() if k != 'go_angle'])}  # same as _min_impact()

        def _make_coherent_stochastic_by_support_table(op_set, infos_by_type):
            table = {}
            table_rows = set()
            table_cols = set()
            for typ, infos_by_actedon in infos_by_type.items():
                if typ == ('H',):
                    col = "Coherent"
                    if self.error_type == "stochastic": continue
                elif typ == ('S',):
                    col = "Stochastic"
                    if self.error_type == "hamiltonian": continue
                else: col = "Mixed"

                table_cols.add(col)
                for acted_on, infos in infos_by_actedon.items():
                    total = _total_contrib(typ, op_set, infos)
                    table_rows.add(acted_on)
                    if acted_on not in table: table[acted_on] = {}
                    table[acted_on][col] = total
            return table, tuple(sorted(table_rows, key=lambda t: (len(t),) + t)), tuple(sorted(table_cols))

        def _compute_by_weight_magnitudes(op_set, infos_by_type):
            mags_by_xtalk_weight = {}  # "weight" of crosstalk is an int.  0 = local
            infos_by_xtalk_weight = {'H': _collections.defaultdict(list),
                                     'S': _collections.defaultdict(list)}
            #target_qubits = set([qi for op in op_set for qi in self.op_to_target_qubits[op]])
            for typ, infos_by_actedon in infos_by_type.items():
                if typ == ('H',):
                    if self.error_type == "stochastic": continue
                elif typ == ('S',):
                    if self.error_type == "hamiltonian": continue
                else:
                    raise ValueError("Invalid type: %s" % str(typ))

                for acted_on, infos in infos_by_actedon.items():
                    # if set(acted_on).issubset(target_qubits):
                    if all([set(acted_on).issubset(self.op_to_target_qubits[op]) for op in op_set]):
                        infos_by_xtalk_weight[typ[0]][0].extend(infos)  # local == weight 0
                    else:
                        infos_by_xtalk_weight[typ[0]][len(acted_on)].extend(infos)  # higher weight

            weights = sorted(set(list(infos_by_xtalk_weight['H'].keys())
                                 + list(infos_by_xtalk_weight['S'].keys())))
            for weight in weights:
                totalH = totalS = 0.0
                items_abbrev = []
                if weight in infos_by_xtalk_weight['H']:
                    totalH = _total_contrib(('H',), op_set, infos_by_xtalk_weight['H'][weight])
                    totalH = min([v for k, v in totalH.items() if k != 'go_angle'])
                    items_abbrev.extend([(info['label_abbrev'], info['coeff'],
                                          _total_contrib(('H',), op_set, [info]))
                                         for info in infos_by_xtalk_weight['H'][weight]])
                if weight in infos_by_xtalk_weight['S']:
                    totalS = _total_contrib(('S',), op_set, infos_by_xtalk_weight['S'][weight])
                    items_abbrev.extend([(info['label_abbrev'], info['coeff'], _total_contrib(('S',), op_set, [info]))
                                         for info in infos_by_xtalk_weight['S'][weight]])

                mags_by_xtalk_weight[weight] = {'total': totalH + totalS,
                                                'items': items_abbrev}  # = list of (abbrev-item-lbl, coeff, contrib)

            return mags_by_xtalk_weight

        def _compute_by_target_magnitudes(op_set, infos_by_type):
            mags_by_target = {}  # "weight" of crosstalk is an int.  0 = local
            infos_by_target = {'H': _collections.defaultdict(list),
                               'S': _collections.defaultdict(list)}
            #target_qubits = set([qi for op in op_set for qi in self.op_to_target_qubits[op]])
            for typ, infos_by_actedon in infos_by_type.items():
                if typ == ('H',):
                    if self.error_type == "stochastic": continue
                elif typ == ('S',):
                    if self.error_type == "hamiltonian": continue
                else:
                    raise ValueError("Invalid type: %s" % str(typ))

                for acted_on, infos in infos_by_actedon.items():
                    infos_by_target[typ[0]][acted_on].extend(infos)

            targets = sorted(set(list(infos_by_target['H'].keys())
                                 + list(infos_by_target['S'].keys())))
            for target in targets:
                totalH = totalS = 0.0
                items_abbrev = []
                if target in infos_by_target['H']:
                    totalH = _total_contrib(('H',), op_set, infos_by_target['H'][target])
                    totalH = min([v for k, v in totalH.items() if k != 'go_angle'])
                    items_abbrev.extend([(info['label_abbrev'], info['coeff'],
                                          _total_contrib(('H',), op_set, [info]))
                                         for info in infos_by_target['H'][target]])
                if target in infos_by_target['S']:
                    totalS = _total_contrib(('S',), op_set, infos_by_target['S'][target])
                    items_abbrev.extend([(info['label_abbrev'], info['coeff'], _total_contrib(('S',), op_set, [info]))
                                         for info in infos_by_target['S'][target]])

                mags_by_target[target] = {'total': totalH + totalS,
                                          'items': items_abbrev}  # = list of (abbrev-item-lbl, coeff, contrib)

            return mags_by_target

        def _make_long_table(op_set, infos_by_type):
            table = {}
            table_rows = []
            table_cols = ('Coefficient', 'Total', 'Raw Label')
            for typ, infos_by_actedon in infos_by_type.items():
                if typ == ('H',) and self.error_type == "stochastic": continue
                if typ == ('S',) and self.error_type == "hamiltonian": continue
                for acted_on, infos in infos_by_actedon.items():
                    for info in infos:
                        total = _total_contrib(typ, op_set, [info])
                        table_rows.append(info['label'])
                        table[info['label']] = {'Coefficient': info['coeff'],
                                                'Raw Label': info['label_raw'],
                                                'Total': total}
            return table, tuple(table_rows), table_cols

        def _make_abbrev_table(op_set, infos_by_type):
            abbrev_label_coeff_list = []
            for typ, infos_by_actedon in infos_by_type.items():
                if typ == ('H',) and self.error_type == "stochastic": continue
                if typ == ('S',) and self.error_type == "hamiltonian": continue
                for acted_on, infos in infos_by_actedon.items():
                    for info in infos:
                        total = _total_contrib(typ, op_set, [info])
                        abbrev_label_coeff_list.append((info['label_abbrev'], info['coeff'], total))
            return abbrev_label_coeff_list

        def _is_dependent(infos_by_type):
            for typ, infos_by_actedon in infos_by_type.items():
                for acted_on, infos in infos_by_actedon.items():
                    for info in infos:
                        if info['dependent'] is False:
                            return False
            return True

        self.op_set_info = {}
        for op_set, op_fogis_by_type in self.bins.items():

            all_H_infos = [info for acted_on, infos in op_fogis_by_type.get(('H',), {}).items() for info in infos]
            all_S_infos = [info for acted_on, infos in op_fogis_by_type.get(('S',), {}).items() for info in infos]

            self.op_set_info[op_set] = {
                'Coherent': _total_contrib(('H',), op_set, all_H_infos),
                'Stochastic': _total_contrib(('S',), op_set, all_S_infos),
                'table': _make_coherent_stochastic_by_support_table(op_set, op_fogis_by_type),
                'longtable': _make_long_table(op_set, op_fogis_by_type),
                'abbrevtable': _make_abbrev_table(op_set, op_fogis_by_type),
                'byweight': _compute_by_weight_magnitudes(op_set, op_fogis_by_type),
                'bytarget': _compute_by_target_magnitudes(op_set, op_fogis_by_type),
                'dependent': _is_dependent(op_fogis_by_type),
                'children': None  # FUTURE?
            }
            assert(('H', 'S') not in op_fogis_by_type)

    def _create_binned_data(self):

        op_labels = self.fogi_info['primitive_op_labels']
        nHam = self.fogi_info['ham_vecs'].shape[1]
        #op_to_target_qubits = {op_label: op_label.sslbls for op_label in op_labels}
        pauli_bases = {}

        def _create_elemgen_info(ordered_op_labels, elemgen_labels_by_op):
            k = 0; info = {}
            for op_label in ordered_op_labels:
                for eglabel in elemgen_labels_by_op[op_label]:
                    nq = len(eglabel[1])
                    if nq not in pauli_bases: pauli_bases[nq] = _Basis.cast('pp', 4**nq)

                    info[k] = {
                        'type': eglabel[0],
                        'qubits': set([i for bel_lbl in eglabel[1:] for i, char in enumerate(bel_lbl) if char != 'I']),
                        'op': op_label,
                        'eglabel': eglabel,
                        'basismx': pauli_bases[nq][eglabel[1]] if (len(eglabel[1:]) == 1) else None
                    }
                    k += 1
            return info

        ham_elemgen_info = _create_elemgen_info(op_labels, self.fogi_info['ham_elgen_labels_by_op'])
        other_elemgen_info = _create_elemgen_info(op_labels, self.fogi_info['other_elgen_labels_by_op'])

        bins = {}
        dependent_indices = set(self.fogi_info['dependent_vec_indices'])
        for i, coeff in enumerate(self.fogi_coeffs):
            vec = self.fogi_info['ham_vecs'][:, i] if i < nHam else self.fogi_info['other_vecs'][:, i - nHam]
            label = self.fogi_info['ham_fogi_labels'][i] if i < nHam else self.fogi_info['other_fogi_labels'][i - nHam]
            label_raw = self.fogi_info['ham_fogi_labels_raw'][i] if i < nHam else \
                self.fogi_info['other_fogi_labels_raw'][i - nHam]
            label_abbrev = self.fogi_info['ham_fogi_labels_abbrev'][i] if i < nHam else \
                self.fogi_info['other_fogi_labels_abbrev'][i - nHam]
            gauge_dir = self.fogi_info['ham_fogi_gauge_directions'][i] if i < nHam else \
                self.fogi_info['other_fogi_gauge_directions'][i - nHam]

            elemgen_info = ham_elemgen_info if i < nHam else other_elemgen_info
            present_elgen_indices = _np.where(_np.abs(vec) > 1e-5)[0]

            ops_involved = set(); qubits_acted_upon = set(); types = set(); basismx = None
            for k in present_elgen_indices:
                k_info = elemgen_info[k]
                ops_involved.add(k_info['op'])
                qubits_acted_upon.update(k_info['qubits'])
                types.add(k_info['type'])
                basismx = k_info['basismx'] if (basismx is None) else basismx + k_info['basismx']  # ROBIN?

            info = {'ops': ops_involved,
                    'types': types,
                    'qubits': qubits_acted_upon,
                    'coeff': coeff,
                    'to_add': abs(coeff) if ('S' in types) else abs(coeff) * basismx,
                    'label': label,
                    'label_raw': label_raw,
                    'label_abbrev': label_abbrev,
                    'dependent': bool(i in dependent_indices),
                    'gauge_dir': gauge_dir,
                    'fogi_vec': vec
                    }
            ops_involved = tuple(sorted(ops_involved))
            types = tuple(sorted(types))
            qubits_acted_upon = tuple(sorted(qubits_acted_upon))
            if ops_involved not in bins: bins[ops_involved] = {}
            if types not in bins[ops_involved]: bins[ops_involved][types] = {}
            if qubits_acted_upon not in bins[ops_involved][types]: bins[ops_involved][types][qubits_acted_upon] = []
            bins[ops_involved][types][qubits_acted_upon].append(info)

        return bins

    def render_grid(self, filename, all_edges=True, physics=False):
        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']

        #import bpdb; bpdb.set_trace()
        target_qubit_groups = tuple(sorted(set(self.op_to_target_qubits.values())))
        groupids = {op_label: target_qubit_groups.index(self.op_to_target_qubits[op_label])
                    for op_label in op_labels}

        y = 0
        increment = 200  # HARDCODED!
        group_yvals = {}
        for grp in target_qubit_groups:
            if len(grp) != 1: continue
            group_yvals[grp] = y; y += increment
        for grp in target_qubit_groups:
            if len(grp) == 1: continue
            group_yvals[grp] = _np.mean([group_yvals[(i,)] for i in grp])

        group_member_counts = {size: _collections.defaultdict(lambda: 0) for size in set(map(len, target_qubit_groups))}
        for op_label in op_labels:
            target_qubits = self.op_to_target_qubits[op_label]
            group_member_counts[len(target_qubits)][target_qubits] += 1
        max_group_member_counts = {size: max(member_counts.values())
                                   for size, member_counts in group_member_counts.items()}
        cum_size = {}; x = 0
        for size in sorted(max_group_member_counts.keys()):
            cum_size[size] = x
            x += max_group_member_counts[size]

        #relational_node_groupid = len(target_qubit_groups)

        node_js_lines = []
        edge_js_lines = []
        table_html = {}
        long_table_html = {}
        existing_pts = []

        def _node_color(value):
            if value < 1e-3: return "green"
            if value < 1e-2: return "yellow"
            if value < 1e-1: return "orange"
            return "red"

        def _make_table(table_info, rowlbl, title):
            table_dict, table_rows, table_cols = table_info
            html = "<table><thead><tr><th colspan=%d>%s</th></tr>\n" % (len(table_cols) + 1, title)
            html += ("<tr><th>%s<th>" % rowlbl) + "</th><th>".join(table_cols) + "</th></tr></thead><tbody>\n"
            for row in table_rows:
                table_row_text = []
                for col in table_cols:
                    val = table_dict[row][col]
                    if _np.iscomplex(val): val = _np.real_if_close(val)
                    if _np.isreal(val) or _np.iscomplex(val):
                        if abs(val) < 1e-6: val = 0.0
                        if _np.isreal(val): table_row_text.append("%.3g" % val.real)
                        else: table_row_text.append("%.3g + %.3gj" % (val.real, val.imag))
                    else: table_row_text.append(str(val))
                html += "<tr><th>" + str(row) + "</th><td>" + "</td><td>".join(table_row_text) + "</td></tr>\n"
            return html + "</tbody></table>"

        #process local quantities
        node_ids = {}; node_locations = {}; next_node_id = 0
        next_xval_by_group = {grp: cum_size[len(grp)] * increment for grp in target_qubit_groups}
        for op_set, info in op_set_info.items():
            if len(op_set) != 1: continue
            # create a gate-node in the graph
            target_qubits = self.op_to_target_qubits[op_set[0]]
            node_js_lines.append('{ id: %d, label: "%s", group: %d, title: "%s", x: %d, y: %d, color: "%s"}' %
                                 (next_node_id, str(op_set[0]), groupids[op_set[0]],
                                  "Coherent: %.3g<br>Stochastic: %.3g" % (info['Coherent'], info['Stochastic']),
                                  next_xval_by_group[target_qubits], group_yvals[target_qubits],
                                  _node_color(info['Coherent'] + info['Stochastic'])))
            table_html[next_node_id] = _make_table(info['table'], "Qubits", "Local errors on %s" % str(op_set[0]))
            long_table_html[next_node_id] = _make_table(info['longtable'], "Label",
                                                        "FOGI quantities for %s" % str(op_set[0]))

            node_locations[op_set[0]] = (next_xval_by_group[target_qubits], group_yvals[target_qubits])
            existing_pts.append(node_locations[op_set[0]])
            next_xval_by_group[target_qubits] += increment
            node_ids[op_set[0]] = next_node_id
            next_node_id += 1

        #process relational quantities
        relational_values = [(op_set, info['Coherent'][0] + info['Stochastic'])
                             for op_set, info in op_set_info.items() if len(op_set) != 1]
        sorted_relational_opsets = sorted(relational_values, key=lambda x: x[1], reverse=True)  # in decreasing "value"

        globs = {op_label: i for i, op_label in enumerate(op_labels)}  # each node is in it's own glob
        for op_set, val in sorted_relational_opsets:
            info = op_set_info[op_set]
            if len(op_set) == 1: continue  # already processed
            if abs(val) < 1e-6: continue  # prune edges that are small
            if not all_edges:
                if all([globs[op_label] == globs[op_set[0]] for op_label in op_set]): continue  # already connected!

            # create a relational node in the graph
            avg_x = int(_np.mean([node_locations[op_label][0] for op_label in op_set]))
            avg_y = int(_np.mean([node_locations[op_label][1] for op_label in op_set]))
            while True:
                for x, y in existing_pts:
                    if _np.sqrt((avg_x - x)**2 + (avg_y - y)**2) < increment / 10:
                        if abs(avg_x - x) >= abs(avg_y - y): avg_y += increment / 3
                        else: avg_x += increment / 3
                        break
                else:
                    break

            #val = info['Coherent'] + info['Stochastic']  # computed above
            node_js_lines.append('{ id: %d, group: "%s", title: "%s", x: %d, y: %d, color: "%s" }' %
                                 (next_node_id, "relational",
                                  "Coherent: %s<br>Stochastic: %.3g" % (str(info['Coherent']), info['Stochastic']),
                                  avg_x, avg_y, _node_color(val)))
            existing_pts.append((avg_x, avg_y))
            table_html[next_node_id] = _make_table(info['table'], "Qubits",
                                                   "Relational errors between " + ", ".join(map(str, op_set)))
            long_table_html[next_node_id] = _make_table(info['longtable'], "Label",
                                                        "FOGI quantities for " + ", ".join(map(str, op_set)))

            #link to gate-nodes
            for op_label in op_set:
                edge_js_lines.append('{ from: %d, to: %d, value: %.4f, dashes: %s}' % (
                    next_node_id, node_ids[op_label], val, 'true' if info['dependent'] else 'false'))
            next_node_id += 1

            globs_to_reassign = set([globs[op_label] for op_label in op_set])
            new_glob_id = globs_to_reassign.pop()
            for op_label in globs.keys():
                if globs[op_label] in globs_to_reassign:
                    globs[op_label] = new_glob_id

        all_table_html = ""
        for node_id, html in table_html.items():
            all_table_html += ("<div class='infotable dataTable' id='%d'>\n" % node_id) + html + "</div>\n"

        all_long_table_html = ""
        for node_id, html in long_table_html.items():
            all_long_table_html += ("<div class='infotable dataTable' id='long%d'>\n" % node_id) + html + "</div>\n"

        s = _template.format(**{'node_js': ",\n".join(node_js_lines),
                                'edge_js': ",\n".join(edge_js_lines),
                                'table_html': all_table_html,
                                'long_table_html': all_long_table_html,
                                'physics': 'true' if physics else "false",
                                'springlength': 100,
                                'beforeDrawingCalls': ''
                                })
        with open(filename, 'w') as f:
            f.write(s)

    def render_circle(self, filename, physics=True,
                      numerical_labels=False, edge_threshold=1e-6):

        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']

        #Group ops based on target qubits
        target_qubit_groups = tuple(sorted(set(self.op_to_target_qubits.values())))
        groupids = {op_label: target_qubit_groups.index(self.op_to_target_qubits[op_label])
                    for op_label in op_labels}
        groups = {group_id: [] for group_id in range(len(target_qubit_groups))}
        for op_label in op_labels:
            groups[groupids[op_label]].append(op_label)

        # Position ops around a circle with more space between groups)
        nPositions = len(op_labels) + len(groups)  # includes gaps between groups
        r0 = 40 * nPositions  # heuristic
        theta = 0; dtheta = 2 * _np.pi / nPositions
        springlength = r0 * dtheta / 3.0  # heuristic
        polar_positions = {}; group_theta_ranges = {}
        for group_id, ops_in_group in groups.items():
            theta_begin = theta
            for op_label in ops_in_group:
                polar_positions[op_label] = (r0, theta)
                theta += dtheta
            group_theta_ranges[group_id] = (theta_begin - dtheta / 3, theta - dtheta + dtheta / 3)
            theta += dtheta

        # Background wedges
        beforeDrawingCalls = ""
        for group_id, (theta_begin, theta_end) in group_theta_ranges.items():
            target_qubits = target_qubit_groups[group_id]
            txt = ("Qubit %d" % target_qubits[0]) if len(target_qubits) == 1 \
                else ("Qubits " + ", ".join(map(str, target_qubits)))
            txt_theta = (theta_begin + theta_end) / 2
            x, y = (r0 + 50) * _np.cos(txt_theta), (r0 + 50) * _np.sin(txt_theta)  # text location
            txt_angle = (txt_theta + _np.pi) if (0 <= txt_theta <= _np.pi) else txt_theta
            beforeDrawingCalls += ('ctx.beginPath();\n'
                                   'ctx.arc(0, 0, %f, %f, %f, false);\n'
                                   'ctx.lineTo(0, 0);\n'
                                   'ctx.closePath();\n'
                                   'ctx.fill();\n'
                                   'ctx.save();\n'
                                   'ctx.translate(%f, %f);\n'
                                   'ctx.rotate(Math.PI / 2 + %f);\n'
                                   'ctx.fillText("%s", 0, 0);\n'
                                   'ctx.restore();\n') % (
                                       r0 + 30, theta_begin, theta_end, x, y, txt_angle, txt)

        node_js_lines = []
        edge_js_lines = []
        table_html = {}
        long_table_html = {}

        def _node_color(value):
            MAX_POWER = 5
            r, g, b, a = _cmap(-_np.log10(max(value, 10**(-MAX_POWER))) / MAX_POWER)  # or larger
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

        def _dstr(d, joinstr="<br>"):  # dict-to-string formatting function
            if len(d) == 1: return "%.3g" % next(iter(d.values()))
            return joinstr.join(["%s: %.3g" % (k, v) for k, v in d.items()])

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

        #process local quantities
        node_ids = {}; node_locations = {}; next_node_id = 0
        for op_set, info in op_set_info.items():
            if len(op_set) != 1: continue
            # create a gate-node in the graph
            r, theta = polar_positions[op_set[0]]

            label = str(op_set[0])
            if self.error_type == 'both':
                title = "Coherent: %.3g<br>Stochastic: %.3g" % (info['Coherent']['errgen_angle'], info['Stochastic'])
                back_color = _node_color(info['Stochastic'])
                border_color = _node_color(info['Coherent']['errgen_angle'])
                if numerical_labels: label += "\\n<i>H: %.3g S: %.3g</i>" \
                   % (info['Coherent']['errgen_angle'], info['Stochastic'])
            elif self.error_type == 'hamiltonian':
                title = "%.3g" % info['Coherent']['errgen_angle']
                back_color = border_color = _node_color(info['Coherent']['errgen_angle'])
                if numerical_labels: label += "\\n<i>%.3g</i>" % info['Coherent']['errgen_angle']
            elif self.error_type == 'stochastic':
                title = "%.3g" % info['Stochastic']
                back_color = border_color = _node_color(info['Stochastic'])
                if numerical_labels: label += "\\n<i>%.3g</i>" % info['Stochastic']
            else:
                raise ValueError("Invalid error_type: %s" % self.error_type)

            node_js_lines.append(('{id: %d, label: "%s", group: %d, title: "%s", x: %d, y: %d,'
                                  'color: {background: "%s", border: "%s"}, fixed: %s}') %
                                 (next_node_id, label, groupids[op_set[0]],
                                  title, int(r * _np.cos(theta)), int(r * _np.sin(theta)), back_color, border_color,
                                  'true' if physics else 'false'))
            table_html[next_node_id] = _make_table(info['table'], "Qubits", "Local errors on %s" % str(op_set[0]))
            long_table_html[next_node_id] = _make_table(info['longtable'],
                                                        "Label", "FOGI quantities for %s" % str(op_set[0]))
            node_locations[op_set[0]] = int(r * _np.cos(theta)), int(r * _np.sin(theta))
            node_ids[op_set[0]] = next_node_id
            next_node_id += 1

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

        def _min_impact(coh_dict):
            return min([v for k, v in coh_dict.items() if k != 'go_angle'])

        #place the longest nPositions linking nodes in the center; place the rest on the periphery
        for i, (op_set, _) in enumerate(relational_opsets_by_distance):
            info = op_set_info[op_set]

            if self.error_type == 'both' and abs(_min_impact(info['Coherent'])) < 1e-6 \
               and abs(info['Stochastic']) < edge_threshold: continue  # prune edges that are small
            elif self.error_type == 'hamiltonian' and abs(_min_impact(info['Coherent'])) < edge_threshold: continue
            elif self.error_type == 'stochastic' and abs(info['Stochastic']) < edge_threshold: continue

            # create a relational node in the graph
            if i < nPositions:  # place node in the middle (just average coords
                x = int(_np.mean([node_locations[op_label][0] for op_label in op_set]))
                y = int(_np.mean([node_locations[op_label][1] for op_label in op_set]))
            else:  # place node along periphery
                r = r0 * 1.1  # push outward from ring of operations
                theta = _np.arctan2(_np.sum([node_locations[op_label][1] for op_label in op_set]),
                                    _np.sum([node_locations[op_label][0] for op_label in op_set]))
                x, y = int(r * _np.cos(theta)), int(r * _np.sin(theta))

            label = ""
            if self.error_type == 'both':
                title = "Coherent: %s<br>Stochastic: %.3g" % (_dstr(info['Coherent']), info['Stochastic'])
                back_color, border_color = _node_color(info['Stochastic']), _node_color(_min_impact(info['Coherent']))
                if numerical_labels: label += "H: %s S: %.3g" % (_dstr(info['Coherent'], r'\n'), info['Stochastic'])
            elif self.error_type == 'hamiltonian':
                title = "%s" % _dstr(info['Coherent'])
                back_color = border_color = _node_color(_min_impact(info['Coherent']))
                if numerical_labels: label += "%s" % _dstr(info['Coherent'], r'\n')
            elif self.error_type == 'stochastic':
                title = "%.3g" % info['Stochastic']
                back_color = border_color = _node_color(info['Stochastic'])
                if numerical_labels: label += "%.3g" % info['Stochastic']
            else:
                raise ValueError("Invalid error_type: %s" % self.error_type)

            node_js_lines.append(('{ id: %d, label: "%s", group: "%s", title: "%s", x: %d, y: %d,'
                                  'color: {background: "%s", border: "%s"}, font: {size: %d, '
                                  'strokeWidth: 3, strokeColor: "white"} }') %
                                 (next_node_id, label, "relational", title, x, y, back_color, border_color, 12))
            table_html[next_node_id] = _make_table(info['table'], "Qubits",
                                                   "Relational errors between " + ", ".join(map(str, op_set)))
            long_table_html[next_node_id] = _make_table(info['longtable'], "Label",
                                                        "FOGI quantities for " + ", ".join(map(str, op_set)))
            #link to gate-nodes
            for op_label in op_set:
                if self.error_type == 'both':
                    val = info['Stochastic'] + _min_impact(info['Coherent'])
                elif self.error_type == 'hamiltonian':
                    val = _min_impact(info['Coherent'])
                elif self.error_type == 'stochastic':
                    val = info['Stochastic']

                edge_js_lines.append('{ from: %d, to: %d, value: %.4f, dashes: %s }' % (
                    next_node_id, node_ids[op_label], val, 'true' if info['dependent'] else 'false'))
            next_node_id += 1

        all_table_html = ""
        for node_id, html in table_html.items():
            all_table_html += ("<div class='infotable dataTable' id='%d'>\n" % node_id) + html + "</div>\n"

        all_long_table_html = ""
        for node_id, html in long_table_html.items():
            all_long_table_html += ("<div class='infotable dataTable' id='long%d'>\n" % node_id) + html + "</div>\n"

        s = _template.format(**{'node_js': ",\n".join(node_js_lines),
                                'edge_js': ",\n".join(edge_js_lines),
                                'table_html': all_table_html,
                                'long_table_html': all_long_table_html,
                                'physics': 'true' if physics else "false",
                                'springlength': springlength,
                                'beforeDrawingCalls': beforeDrawingCalls})
        with open(filename, 'w') as f:
            f.write(s)

    def plot_heatmap(self, include_xtalk=False, fig_base=5):
        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']

        data = _np.ones((len(op_labels), len(op_labels)), 'd') * _np.nan
        op_label_lookup = {lbl: i for i, lbl in enumerate(op_labels)}
        xtalk_datas = []

        def _min_impact(coh_dict):
            if 'min_impact' in coh_dict: return coh_dict['min_impact']
            return min([v for k, v in coh_dict.items() if k != 'go_angle'])

        additional = 0
        for op_set, info in op_set_info.items():
            if self.error_type == "hamiltonian": val = _min_impact(info['Coherent'])
            elif self.error_type == "stochastic": val = info['Stochastic']
            elif self.error_type == "both": val = info['Stochastic'] + _min_impact(info['Coherent'])
            else: raise ValueError("Invalid error_type: %s" % self.error_type)

            if len(op_set) > 2:
                additional += val
                continue

            if len(op_set) == 2:
                i, j = op_label_lookup[op_set[0]], op_label_lookup[op_set[1]]
                if i > j: i, j = j, i
            else:
                i = j = op_label_lookup[op_set[0]]

            assert(_np.isnan(data[i, j]))
            data[i, j] = val
            if include_xtalk:
                for weight, xtalk_dict in info['byweight'].items():
                    xtalk_val = xtalk_dict['total']
                    while len(xtalk_datas) < weight + 1:
                        xtalk_datas.append(_np.ones((len(op_labels), len(op_labels)), 'd') * _np.nan)
                    assert(_np.isnan(xtalk_datas[weight][i, j]))
                    xtalk_datas[weight][i, j] = xtalk_val

        import matplotlib
        import matplotlib.pyplot as plt

        min_color, max_color = 0, _np.nanmax(data)
        fig, ax_tuple = plt.subplots(1, 1 + len(xtalk_datas), figsize=(fig_base * (1 + len(xtalk_datas)), fig_base))
        if len(xtalk_datas) == 0: ax_tuple = (ax_tuple,)  # because matplotlib is too clever
        fig.suptitle("%s-type errors on gates and between gate pairs (missing %.3g)" % (self.error_type, additional))
        axis_labels = [str(lbl) for lbl in op_labels]

        for plot_data, title, ax in zip([data] + xtalk_datas,
                                        ["Total"] + [("Local" if i == 0 else "Weight-%d crosstalk" % i)
                                                     for i in range(len(xtalk_datas))],
                                        ax_tuple):
            im = ax.imshow(plot_data, cmap="Reds")
            im.set_clim(min_color, max_color)

            # We want to show all ticks...
            ax.set_xticks(_np.arange(len(axis_labels)))
            ax.set_yticks(_np.arange(len(axis_labels)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(axis_labels)
            ax.set_yticklabels(axis_labels)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Normalize the threshold to the images color range.
            threshold = im.norm(_np.nanmax(plot_data)) / 2.

            # Loop over data dimensions and create text annotations.
            textcolors = ['black', 'white']
            for i in range(len(axis_labels)):
                for j in range(len(axis_labels)):
                    if not _np.isnan(plot_data[i, j]):
                        ax.text(j, i, "%.1g" % plot_data[i, j],
                                ha="center", va="center",
                                color=textcolors[int(im.norm(plot_data[i, j]) > threshold)])

            ax.set_title(title)

        fig.tight_layout()
        plt.show()
        return fig

    def create_detail_table(self, filename, mode='individual_terms'):
        from .table import ReportTable as _ReportTable

        assert(mode in ('individual_terms', 'by_support'))
        assert(self.error_type in ('hamiltonian', 'stochastic')), \
            "Detail tables must have error_type of 'hamiltonian' or 'stochastic'!"  # b/c abbrev. labels drop 'H' & 'S'
        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']

        op_label_lookup = {lbl: i for i, lbl in enumerate(op_labels)}
        cell_tabulars = {}

        def _dstr(d, op_set):  # dict-to-string formatting function
            if not isinstance(d, dict):
                if abs(d) < 1e-6: d = 0.0
                return ("%.3g" % d) + " & " * len(op_set)
            if len(d) == 1:
                v = next(iter(d.values()))
                if abs(v) < 1e-6: v = 0.0
                return "%.3g" % v + " & " * len(op_set)
            #if len(op_set) == 1: return "%.3g" % d['errgen_angle']  # covered by above case
            return "%.3g & " % d['go_angle'] + " & ".join(["%.3g" % d[op] for op in op_set])

        additional = 0
        for op_set, info in op_set_info.items():
            if mode == 'individual_terms':
                tabular = "\\begin{tabular}{@{}%s@{}}" % ('c' * (3 + len(op_set)))  \
                    + " \\\\ ".join(["%s & %.3g & %s" % (lbl, abs(coeff) if abs(coeff) > 1e-6 else 0.0,
                                                         _dstr(total, op_set))
                                     for lbl, coeff, total in info['abbrevtable']]) \
                    + r"\end{tabular}"
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

    def plot_multiscale_grid(self, detail_level=0, figsize=5, outfile=None, spacing=0.05, nudge=0.125,
                             cell_fontsize=10, axes_fontsize=10, qty_key='bytarget'):
        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']
        nOps = len(op_labels)

        op_label_lookup = {lbl: i for i, lbl in enumerate(op_labels)}
        totals = _np.ones((len(op_labels), len(op_labels)), 'd') * _np.nan
        by_qty_totals = {}
        by_qty_items = {}

        def _min_impact(coh_dict):
            if 'min_impact' in coh_dict: return coh_dict['min_impact']
            return min([v for k, v in coh_dict.items() if k != 'go_angle'])

        additional = 0
        for op_set, info in op_set_info.items():

            if len(op_set) == 2:
                i, j = op_label_lookup[op_set[0]], op_label_lookup[op_set[1]]
                if i > j: i, j = j, i
            else:
                i = j = op_label_lookup[op_set[0]]

            #Total value/contribution for entire "box"
            if self.error_type == "hamiltonian": val = _min_impact(info['Coherent'])
            elif self.error_type == "stochastic": val = info['Stochastic']
            elif self.error_type == "both": val = info['Stochastic'] + _min_impact(info['Coherent'])
            else: raise ValueError("Invalid error_type: %s" % self.error_type)

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
        axes.set_title("%s-type errors on gates and between gate pairs (missing %.3g)" % (self.error_type, additional))
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
                        nm = ", ".join(["Q%d" % (qlbl + 1) for qlbl in qty])
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
                        if isinstance(contrib, dict):
                            assert(len(contrib) == 1)
                            contrib = next(iter(contrib.values()))
                        if abs(contrib) < 1e-6: contrib = 0.0

                        y3 -= 1
                        val_color = cm(contrib / val_max)[0:3]  # drop alpha
                        rect = patches.Rectangle((x, y3), side, 1.0, linewidth=0, edgecolor='k',
                                                 facecolor=val_color)
                        ax.add_patch(rect)
                        ax.text(x + side / 2, y3 + 1 / 2, "%s = %.1f%%" % (lbl, contrib * 100),
                                ha="center", va="center", color=textcolors[int(contrib > val_threshold)],
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
              size: 20,
          }},
          color: {{highlight: {{ background: "white", border: "black" }},
                   hover: {{ background: "white", border: "black" }} }},
          borderWidth: 3,
      }},
      edges: {{
          smooth: false,
          length: {springlength},
          color: {{color: "gray", highlight: "black"}},
          scaling: {{min: 1, max: 6}},
          //width: 2,
          //font: {{
          //    size: 8
          //}},
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
