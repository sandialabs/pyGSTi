import stim
import networkx as nx
import pyomo.environ as pe
from itertools import product
import matplotlib.pyplot as plt

# non-relative imports for now
from pygsti.extras.idletomography import idtcore
import collections as _collections
import itertools as _itertools
import time as _time
import warnings as _warnings
import more_itertools as _mi
from itertools import product, permutations
from pygsti.baseobjs import basisconstructors
import numpy as _np
from pygsti.baseobjs import Basis


# Commutator Helper Functions
def commute(mat1, mat2):
    return mat1 @ mat2 + mat2 @ mat1


def anti_commute(mat1, mat2):
    return mat1 @ mat2 - mat2 @ mat1


# Hamiltonian Error Generator
# returns output of applying error gen in choi unit form
# input is state, output is state
def hamiltonian_error_generator(initial_state, indexed_pauli, identity):
    return 1 * (
        -1j * indexed_pauli @ initial_state @ identity
        + 1j * identity @ initial_state @ indexed_pauli
    )


# Stochastic Error Generator
def stochastic_error_generator(initial_state, indexed_pauli, identity):
    return 1 * (
        indexed_pauli @ initial_state @ indexed_pauli
        - identity @ initial_state @ identity
    )


def coverage_edge_exists(error_gen_type, pauli_index, prep_string, meas_string):
    # print(f"{error_gen_type}")
    # print(f"{pauli_index}")
    # print(f"{prep_string}")
    # print(f"{meas_string}")
    if error_gen_type == "h":
        prep_ham_idx_comm = idtcore.half_pauli_comm(pauli_index, prep_string)
        # The paulis are trace orthonormal, so we only get a non-zero value
        # if meas_overall ~= prep_ham_idx_comm (i.e. up to the overall sign/phase).
        if prep_ham_idx_comm == 0:
            return False
        if idtcore.is_unsigned_pauli_equiv(meas_string, prep_ham_idx_comm):
            return True
    elif error_gen_type == "s":
        if not pauli_index.commutes(prep_string):
            if idtcore.is_unsigned_pauli_equiv(meas_string, prep_string):
                return True
    return False


def alt_coverage_edge_exists(error_gen_type, pauli_index, prep_string, meas_string):
    if error_gen_type == "h":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        prep_string_iterator = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )
        t = 0
        for string in prep_string_iterator:
            h = hamiltonian_error_generator(
                string.to_unitary_matrix(endian="little"),
                pauli_index.to_unitary_matrix(endian="little"),
                ident.to_unitary_matrix(endian="little"),
            )
            # want approx non-zero rather than strict
            if _np.any(h):
                h_string = stim.PauliString.from_unitary_matrix(h / 2)
                second_matrix = meas_string * h_string
                # what is the correct coefficient here?
                # t += (1 / 2**num_qubits) * _np.trace(
                t += _np.trace(second_matrix.to_unitary_matrix(endian="little"))
                # print(t)
                if _np.absolute(t) > 0.0001:
                    print()
                    print(
                        f"prep string: {string}, other thing: {stim.PauliString.after(meas_string,h_string.to_tableau(), targets=[i for i in range(len(meas_string))])}, h_ string? {h_string}, pauli index: {pauli_index}, measure string: {meas_string}, coef: {t}"
                    )
                    print()
                    # quit()
                    return True
    elif error_gen_type == "s":
        return False
        s = stochastic_error_generator(
            prep_string.to_unitary_matrix(endian="little"),
            pauli_index.to_unitary_matrix(endian="little"),
            ident.to_unitary_matrix(endian="little"),
        )
        if _np.any(s):
            s_string = stim.PauliString.from_unitary_matrix(s / 2)
            # if s_string == meas_string or s_string == -meas_string:
            if stim.PauliString.commutes(s_string, meas_string):
                return True
    return False


num_qubits = 3
max_weight = 2

HS_index_iterator = stim.PauliString.iter_all(
    num_qubits, min_weight=1, max_weight=max_weight
)

pauli_node_attributes = list([p for p in HS_index_iterator])
#   + [-p for p in iterator_huzzah])

measure_string_iterator = stim.PauliString.iter_all(num_qubits, min_weight=num_qubits)
measure_string_attributes = list([p for p in measure_string_iterator])
prep_string_attributes = measure_string_attributes
prep_meas_pairs = list(product(prep_string_attributes, measure_string_attributes))


# print(prep_meas_pairs)

ident = stim.PauliString(num_qubits)
# prep = prep_meas_pairs[0][0]
# index = pauli_node_attributes[1]
# meas = prep_meas_pairs[0][1]
# print(ident, prep, meas, index)
# h = hamiltonian_error_generator(prep.to_unitary_matrix(endian="little"), index.to_unitary_matrix(endian="little"), ident.to_unitary_matrix(endian="little"))
# print(h)
# print(stim.PauliString.from_unitary_matrix(h/2))
# m = meas.to_unitary_matrix(endian="little")
# print(commute(m,h))
# quit()

test_graph = nx.Graph()
# test_graph.add_nodes_from(enumerate(pauli_node_attributes), pauli_string = pauli_node_attributes, bipartite=1)
for i, j in prep_meas_pairs:
    test_graph.add_node(
        len(test_graph.nodes) + 1, prep_string=i, meas_string=j, bipartite=0
    )

error_gen_classes = "h"
# error_gen_classes = "hs"

for j in error_gen_classes:
    for i in range(len(pauli_node_attributes)):
        test_graph.add_node(
            len(test_graph.nodes) + 1,
            error_gen_class=j,
            pauli_index=pauli_node_attributes[i],
            bipartite=1,
        )

# print(test_graph.nodes[88])
# print([test_graph.nodes[node] for node in test_graph.nodes])
# quit()
bipartite_identifier = nx.get_node_attributes(test_graph, "bipartite")
# hey rewrite this to not be stupid.  or at least less stupid.
bipartite_pairs = [
    (k1, k2)
    for k1 in bipartite_identifier.keys()
    if bipartite_identifier[k1] == 0
    for k2 in bipartite_identifier.keys()
    if bipartite_identifier[k2] == 1
]

for pair in bipartite_pairs:
    if alt_coverage_edge_exists(
        test_graph.nodes[pair[1]]["error_gen_class"],
        test_graph.nodes[pair[1]]["pauli_index"],
        test_graph.nodes[pair[0]]["prep_string"],
        test_graph.nodes[pair[0]]["meas_string"],
    ):
        test_graph.add_edge(pair[0], pair[1])


# print(list(test_graph.nodes[n] for n in test_graph.nodes))
# print(list(edge for edge in test_graph.edges if edge[1]==87))
# quit()
# print(test_graph.nodes[88])
labels = {n: "" for n in test_graph.nodes}
pos = {n: (0, 0) for n in test_graph.nodes}
x_pos_err = 0
x_pos_exp = 0
# save_me = [2]
for n in test_graph.nodes:
    if test_graph.nodes[n].get("pauli_index"):
        labels[n] = (
            str(test_graph.nodes[n].get("error_gen_class"))
            + "_"
            + str(test_graph.nodes[n].get("pauli_index"))
        )
        pos[n] = (x_pos_err, 1)
        x_pos_err += 7000
    #         save_me.append(n)
    else:
        labels[n] = (
            str(test_graph.nodes[n]["prep_string"])
            + " / "
            + str(test_graph.nodes[n]["meas_string"])
        )
        pos[n] = (x_pos_exp, 0)
        x_pos_exp += 1000

# hxy_subgraph = nx.subgraph(test_graph, save_me)


# x_pos_err = 0
# x_pos_exp = 0
# for n in hxy_subgraph.nodes:
#     if test_graph.nodes[n].get("pauli_index"):
#         labels[n] = str(test_graph.nodes[n].get("error_gen_class")) + "_" + str(test_graph.nodes[n].get("pauli_index"))
#         pos[n] = (x_pos_err, 3)
#         x_pos_err += 2
#     else:
#         labels[n] = str(test_graph.nodes[n]["prep_string"]) + " / " + str(test_graph.nodes[n]["meas_string"])
#         pos[n] = (x_pos_exp, 0)
#         x_pos_exp += 2


# nx.draw(test_graph, nx.kamada_kawai_layout(test_graph))
plt.figure(figsize=(11, 8.5))
nx.draw_networkx_nodes(test_graph, pos, node_size=15)
nx.draw_networkx_edges(test_graph, pos)
nx.draw_networkx_labels(test_graph, pos, labels=labels, font_size=2)


plt.savefig("dum_graf.pdf")
m = pe.ConcreteModel()
m.covering_nodes = [
    n for n in test_graph.nodes if test_graph.nodes[n]["bipartite"] == 0
]
m.error_generator_nodes = [
    n for n in test_graph.nodes if test_graph.nodes[n]["bipartite"] == 1
]
m.edges = test_graph.edges
m.num_qubits = num_qubits
# print(m.edges)
m.experiment_choice = pe.Var(m.covering_nodes, domain=pe.Binary, initialize=0)
m.known_error_generators = pe.Var(
    m.error_generator_nodes, domain=pe.Binary, initialize=0
)
m.information_streams = pe.Var(m.edges, domain=pe.Binary, initialize=0)

# @m.Constraint(m.error_generator_nodes)
# def covering_logic_rule(m,covered_node):
#     return sum(m.experiment_choice[covering_node] for (covering_node,cov_node) in m.edges if cov_node==covered_node) >= m.known_error_generators[covered_node]

# @m.Constraint(m.error_generator_nodes)
# def full_knowledge_rule(m, covered_node):
#     return sum(m.experiment_choice[covering_node] for (covering_node,cov_node) in m.edges if cov_node == covered_node) >= nx.degree(test_graph, covered_node)


@m.Constraint(m.edges)
def error_gen_covering_rule(m, *edge):
    return m.known_error_generators[edge[1]] >= m.information_streams[edge]


@m.Constraint(m.edges)
def experiment_covering_rule(m, *edge):
    return m.experiment_choice[edge[0]] >= m.information_streams[edge]


@m.Constraint(m.error_generator_nodes)
def error_gen_selection_rule(m, node):
    return m.known_error_generators[node] <= sum(
        m.information_streams[edge] for edge in m.edges if edge[1] == node
    )


@m.Constraint(m.covering_nodes)
def experiment_selection_rule(m, node):
    return m.experiment_choice[node] <= sum(
        m.information_streams[edge] for edge in m.edges if edge[0] == node
    )


@m.Constraint(m.covering_nodes)
def saturation_rule(m, covering_node):
    if nx.degree(test_graph, covering_node) == 0:
        return pe.Constraint.Skip
    return (
        sum(m.information_streams[edge] for edge in m.edges if edge[0] == covering_node)
        <= 2**m.num_qubits - 1
    )


@m.Constraint(m.error_generator_nodes)
def full_coverage(m, covered_node):
    return m.known_error_generators[covered_node] >= 1


m.obj = pe.Objective(expr=sum(m.experiment_choice[n] for n in m.covering_nodes))
with open("hahahahano.txt", "w") as f:
    m.pprint(f)
opt = pe.SolverFactory("gurobi")
opt.solve(m, tee=True)
# m.pprint()
print(f"{pe.value(m.obj)}")
import re

info_streams = []
exp_egen_pairs = []
for v in m.component_data_objects(ctype=pe.Var):
    if "exp" in v.name and pe.value(v) >= 0.001:
        print(v.name, pe.value(v))
    if "inf" in v.name and pe.value(v) >= 0.001:
        print(v.name)
        nums = re.findall(r"\d+", v.name)
        info_streams.append((int(nums[0]), int(nums[1])))

for info_stream in info_streams:
    egen_class = (
        test_graph.nodes[info_stream[1]]["error_gen_class"]
        + str(test_graph.nodes[info_stream[1]]["pauli_index"])[1:]
    )
    prep_string = test_graph.nodes[info_stream[0]]["prep_string"]
    meas_string = test_graph.nodes[info_stream[0]]["meas_string"]
    exp_egen_pairs.append(((prep_string, meas_string), egen_class))

for pair in exp_egen_pairs:
    print("(" + str(pair[0][0])[1:] + "," + str(pair[0][1])[1:] + ") -----> " + pair[1])
