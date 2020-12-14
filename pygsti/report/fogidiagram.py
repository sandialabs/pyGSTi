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
import collections as _collections
from ..objects import Basis as _Basis


class FOGIDiagram(object):
    """
    A diagram of the first-order-gauge-invariant (FOGI) quantities of a model.

    This class encapsulates a way of visualizing a model's FOGI quantities.
    """

    def __init__(self, model):
        self.fogi_info = model.fogi_info
        self.fogi_coeffs = model.fogi_errorgen_coefficients_array(normalized_elem_gens=False)
        self.bins = self._create_binned_data()

        def _total_contrib(infos_by_actedon):
            total = None
            for acted_on, infos in infos_by_actedon.items():
                for info in infos:
                    total = info['to_add'] if (total is None) else total + info['to_add']
            return _np.linalg.norm(total)

        def _make_coherent_stochastic_by_support_table(infos_by_type):
            table = {}
            table_rows = set()
            table_cols = set()
            for typ, infos_by_actedon in infos_by_type.items():
                if typ == ('H',): col = "Coherent"
                elif typ == ('S',): col = "Stochastic"
                else: col = "Mixed"

                table_cols.add(col)
                for acted_on, infos in infos_by_actedon.items():
                    total = None
                    for info in infos:
                        total = info['to_add'] if (total is None) else total + info['to_add']
                    table_rows.add(acted_on)
                    if acted_on not in table: table[acted_on] = {}
                    table[acted_on][col] = _np.linalg.norm(total)
            return table, tuple(sorted(table_rows, key=lambda t: (len(t),) + t)), tuple(sorted(table_cols))

        def _make_long_table(infos_by_type):
            table = {}
            table_rows = []
            table_cols = ('Coefficient', 'Raw Label')
            for typ, infos_by_actedon in infos_by_type.items():
                for acted_on, infos in infos_by_actedon.items():
                    for info in infos:
                        table_rows.append(info['label'])
                        table[info['label']] = {'Coefficient': info['coeff'],
                                                'Raw Label': info['label_raw']}
            return table, tuple(table_rows), table_cols

        self.op_set_info = {}
        for op_set, op_fogis_by_type in self.bins.items():
            self.op_set_info[op_set] = {
                'Coherent': _total_contrib(op_fogis_by_type[('H',)]) if (('H',) in op_fogis_by_type) else 0.0,
                'Stochastic': _total_contrib(op_fogis_by_type[('S',)]) if (('S',) in op_fogis_by_type) else 0.0,
                'table': _make_coherent_stochastic_by_support_table(op_fogis_by_type),
                'longtable': _make_long_table(op_fogis_by_type),
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
        for i, coeff in enumerate(self.fogi_coeffs):
            vec = self.fogi_info['ham_vecs'][:, i] if i < nHam else self.fogi_info['other_vecs'][:, i - nHam]
            label = self.fogi_info['ham_fogi_labels'][i] if i < nHam else self.fogi_info['other_fogi_labels'][i - nHam]
            label_raw = self.fogi_info['ham_fogi_labels_raw'][i] if i < nHam else \
                self.fogi_info['other_fogi_labels_raw'][i - nHam]
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
                    'label_raw': label_raw
                    }
            ops_involved = tuple(sorted(ops_involved))
            types = tuple(sorted(types))
            qubits_acted_upon = tuple(sorted(qubits_acted_upon))
            if ops_involved not in bins: bins[ops_involved] = {}
            if types not in bins[ops_involved]: bins[ops_involved][types] = {}
            if qubits_acted_upon not in bins[ops_involved][types]: bins[ops_involved][types][qubits_acted_upon] = []
            bins[ops_involved][types][qubits_acted_upon].append(info)

        return bins

    def render_grid(self, filename, op_to_target_qubits=None, all_edges=True, physics=False):
        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']

        if op_to_target_qubits is None:
            all_qubits = set()
            for op_label in op_labels:
                if op_label.sslbls is not None:
                    all_qubits.update(op_label.sslbls)
            all_qubits = tuple(sorted(all_qubits))
            op_to_target_qubits = {op_label: op_label.sslbls if (op_label.sslbls is not None) else all_qubits
                                   for op_label in op_labels}

        #import bpdb; bpdb.set_trace()
        target_qubit_groups = tuple(sorted(set(op_to_target_qubits.values())))
        groupids = {op_label: target_qubit_groups.index(op_to_target_qubits[op_label])
                    for op_label in op_labels}

        y = 0
        increment = 200 #HARDCODED!
        group_yvals = {}
        for grp in target_qubit_groups:
            if len(grp) != 1: continue
            group_yvals[grp] = y; y += increment
        for grp in target_qubit_groups:
            if len(grp) == 1: continue
            group_yvals[grp] = _np.mean([group_yvals[(i,)] for i in grp])

        group_member_counts = {size: _collections.defaultdict(lambda: 0) for size in set(map(len, target_qubit_groups))}
        for op_label in op_labels:
            target_qubits = op_to_target_qubits[op_label]
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
            target_qubits = op_to_target_qubits[op_set[0]]
            node_js_lines.append('{ id: %d, label: "%s", group: %d, title: "%s", x: %d, y: %d, color: "%s"}' %
                                 (next_node_id, str(op_set[0]), groupids[op_set[0]],
                                  "Coherent: %.3g\\nStochastic: %.3g" % (info['Coherent'], info['Stochastic']),
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
        relational_values = [(op_set, info['Coherent'] + info['Stochastic'])
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
                                  "Coherent: %.3g\\nStochastic: %.3g" % (info['Coherent'], info['Stochastic']),
                                  avg_x, avg_y, _node_color(val)))
            existing_pts.append((avg_x, avg_y))
            table_html[next_node_id] = _make_table(info['table'], "Qubits",
                                                   "Relational errors between " + ", ".join(map(str, op_set)))
            long_table_html[next_node_id] = _make_table(info['longtable'], "Label",
                                                        "FOGI quantities for " + ", ".join(map(str, op_set)))

            #link to gate-nodes
            for op_label in op_set:
                edge_js_lines.append('{ from: %d, to: %d, value: %.4f }' % (next_node_id, node_ids[op_label], val))
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

    def render_circle(self, filename, op_to_target_qubits=None, physics=True):
        op_set_info = self.op_set_info
        op_labels = self.fogi_info['primitive_op_labels']

        if op_to_target_qubits is None:
            all_qubits = set()
            for op_label in op_labels:
                if op_label.sslbls is not None:
                    all_qubits.update(op_label.sslbls)
            all_qubits = tuple(sorted(all_qubits))
            op_to_target_qubits = {op_label: op_label.sslbls if (op_label.sslbls is not None) else all_qubits
                                   for op_label in op_labels}

        #Group ops based on target qubits
        target_qubit_groups = tuple(sorted(set(op_to_target_qubits.values())))
        groupids = {op_label: target_qubit_groups.index(op_to_target_qubits[op_label])
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
            x, y = (r0 + 40) * _np.cos(txt_theta), (r0 + 40) * _np.sin(txt_theta)
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
        for op_set, info in op_set_info.items():
            if len(op_set) != 1: continue
            # create a gate-node in the graph
            r, theta = polar_positions[op_set[0]]
            node_js_lines.append(('{id: %d, label: "%s", group: %d, title: "%s", x: %d, y: %d,'
                                  'color: {background: "%s", border: "%s"}, fixed: %s}') %
                                 (next_node_id, str(op_set[0]), groupids[op_set[0]],
                                  "Coherent: %.3g\\nStochastic: %.3g" % (info['Coherent'], info['Stochastic']),
                                  int(r * _np.cos(theta)), int(r * _np.sin(theta)), _node_color(info['Stochastic']),
                                  _node_color(info['Coherent']), 'true' if physics else 'false'))
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

        #place the longest nPositions linking nodes in the center; place the rest on the periphery
        for i, (op_set, _) in enumerate(relational_opsets_by_distance):
            info = op_set_info[op_set]

            if abs(info['Coherent']) < 1e-6 and abs(info['Stochastic']): continue  # prune edges that are small

            # create a relational node in the graph
            if i < nPositions:  # place node in the middle (just average coords
                x = int(_np.mean([node_locations[op_label][0] for op_label in op_set]))
                y = int(_np.mean([node_locations[op_label][1] for op_label in op_set]))
            else:  # place node along periphery
                r = r0 * 1.1  # push outward from ring of operations
                theta = _np.mean([polar_positions[op_label][1] for op_label in op_set])
                x, y = int(r * _np.cos(theta)), int(r * _np.sin(theta))

            node_js_lines.append(('{ id: %d, group: "%s", title: "%s", x: %d, y: %d,'
                                  'color: {background: "%s", border: "%s"} }') %
                                 (next_node_id, "relational",
                                  "Coherent: %.3g\\nStochastic: %.3g" % (info['Coherent'], info['Stochastic']),
                                  x, y, _node_color(info['Stochastic']), _node_color(info['Coherent'])))
            table_html[next_node_id] = _make_table(info['table'], "Qubits",
                                                   "Relational errors between " + ", ".join(map(str, op_set)))
            long_table_html[next_node_id] = _make_table(info['longtable'], "Label",
                                                        "FOGI quantities for " + ", ".join(map(str, op_set)))
            #link to gate-nodes
            for op_label in op_set:
                val = info['Stochastic'] + info['Coherent']
                edge_js_lines.append('{ from: %d, to: %d, value: %.4f }' % (next_node_id, node_ids[op_label], val))
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


_template = """<html>
<head>
    <!-- <link rel="stylesheet" href="pygsti_dataviz.css"> -->
    <!-- <script type="text/javascript" src="vis-network.js"></script> -->
    <!-- <script type="text/javascript" src="jquery-3.2.1.min.js"></script> -->
    <script type="text/javascript" src="vis-network.js"></script>
    <script type="text/javascript" src="jquery-3.2.1.min.js"></script>
    <!-- <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script> -->
    <!-- <script   src="https://code.jquery.com/jquery-3.5.1.min.js"   integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=" crossorigin="anonymous"></script> -->

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
              size: 20,
          }},
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

