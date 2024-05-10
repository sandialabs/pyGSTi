import stim 
import networkx as nx
import pyomo.environ as pe
from itertools import product
import matplotlib.pyplot as plt
#non-relative imports for now
from pygsti.extras.idletomography import idtcore

def coverage_edge_exists(error_gen_type, pauli_index, prep_string, meas_string):
    # print(f"{error_gen_type}")
    # print(f"{pauli_index}")
    # print(f"{prep_string}")
    # print(f"{meas_string}")
    if error_gen_type == "h":
        prep_ham_idx_comm = idtcore.half_pauli_comm(pauli_index, prep_string)
        #The paulis are trace orthonormal, so we only get a non-zero value
        #if meas_overall ~= prep_ham_idx_comm (i.e. up to the overall sign/phase).
        if prep_ham_idx_comm == 0:
            return False
        if idtcore.is_unsigned_pauli_equiv(meas_string, prep_ham_idx_comm):
            return True
    elif error_gen_type == "s":
        if not pauli_index.commutes(prep_string):
            if idtcore.is_unsigned_pauli_equiv(meas_string, prep_string):
                return True
    return False



num_qubits = 3
max_weight = 1

HS_index_iterator = stim.PauliString.iter_all(num_qubits,min_weight=1, max_weight=max_weight)

pauli_node_attributes = list([p for p in HS_index_iterator])
                            #   + [-p for p in iterator_huzzah])

measure_string_iterator = stim.PauliString.iter_all(num_qubits, min_weight=num_qubits)
measure_string_attributes = list([p for p in measure_string_iterator])
prep_string_attributes = measure_string_attributes
prep_meas_pairs = list(product(prep_string_attributes,measure_string_attributes))

test_graph = nx.Graph()
# test_graph.add_nodes_from(enumerate(pauli_node_attributes), pauli_string = pauli_node_attributes, bipartite=1)
for (i,j) in prep_meas_pairs:
    test_graph.add_node(len(test_graph.nodes)+1, prep_string = i, meas_string = j, bipartite = 0)

for j in "hs":
    for i in range(len(pauli_node_attributes)):
        test_graph.add_node(len(test_graph.nodes)+1, error_gen_class = j, pauli_index = pauli_node_attributes[i], bipartite=1)

# print(test_graph.nodes[88])
# print([test_graph.nodes[node] for node in test_graph.nodes])
# quit()
bipartite_identifier = nx.get_node_attributes(test_graph, "bipartite")
# hey rewrite this to not be stupid.  or at least less stupid.
bipartite_pairs = [(k1,k2) for k1 in bipartite_identifier.keys() if bipartite_identifier[k1] ==0 for k2 in bipartite_identifier.keys() if bipartite_identifier[k2] == 1]

for pair in bipartite_pairs:
    if coverage_edge_exists(test_graph.nodes[pair[1]]["error_gen_class"], test_graph.nodes[pair[1]]["pauli_index"], test_graph.nodes[pair[0]]["prep_string"], test_graph.nodes[pair[0]]["meas_string"]):
        test_graph.add_edge(pair[0], pair[1])

# print(list(test_graph.nodes[n] for n in test_graph.nodes))
# print(list(edge for edge in test_graph.edges if edge[1]==87))
# quit()
# print(test_graph.nodes[88])
labels = {n: "" for n in test_graph.nodes}
# pos = {n: (0,0) for n in test_graph.nodes}
# x_pos_err = 0
# x_pos_exp = 0
# save_me = [2]
for n in test_graph.nodes:
    if test_graph.nodes[n].get("pauli_index"):
        labels[n] = str(test_graph.nodes[n].get("error_gen_class")) + "_" + str(test_graph.nodes[n].get("pauli_index"))
#         pos[n] = (x_pos_err, 3)
#         x_pos_err += 1
#         save_me.append(n)
    else:
        labels[n] = str(test_graph.nodes[n]["prep_string"]) + " / " + str(test_graph.nodes[n]["meas_string"])
#         pos[n] = (x_pos_exp, 0)
#         x_pos_exp += 1

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





# # nx.draw(test_graph, nx.kamada_kawai_layout(test_graph))
# plt.figure(figsize=(10,5))
# nx.draw_networkx_nodes(test_graph, pos, node_size = 500, margins=0.01)
# nx.draw_networkx_edges(test_graph, pos)
# nx.draw_networkx_labels(test_graph, pos, labels=labels, font_size=5)


# plt.savefig("dum_graf.png")
m = pe.ConcreteModel()
m.covering_nodes = [n for n in test_graph.nodes if test_graph.nodes[n]["bipartite"] == 0]
m.error_generator_nodes = [n for n in test_graph.nodes if test_graph.nodes[n]["bipartite"] == 1]
m.edges = test_graph.edges
m.num_qubits = num_qubits
# print(m.edges)
m.experiment_choice = pe.Var(m.covering_nodes, domain = pe.Binary, initialize=0)
m.known_error_generators = pe.Var(m.error_generator_nodes, domain = pe.Binary, initialize=0)
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
    return m.known_error_generators[node] <= sum(m.information_streams[edge] for edge in m.edges if edge[1] == node)

@m.Constraint(m.covering_nodes)
def experiment_selection_rule(m, node):
    return m.experiment_choice[node] <= sum(m.information_streams[edge] for edge in m.edges if edge[0] == node)

@m.Constraint(m.covering_nodes)
def saturation_rule(m, covering_node):
    if nx.degree(test_graph,covering_node) == 0:
        return pe.Constraint.Skip
    return sum(m.information_streams[edge] for edge in m.edges if edge[0] == covering_node) <= 2**m.num_qubits - 1  

@m.Constraint(m.error_generator_nodes)
def full_coverage(m, covered_node):
    return m.known_error_generators[covered_node] >= 1

m.obj = pe.Objective(expr=sum(m.experiment_choice[n] for n in m.covering_nodes))
with open("hahahahano.txt","w") as f:
    m.pprint(f)
opt = pe.SolverFactory("gurobi")
opt.solve(m, tee=True)
# m.pprint()
print(f"{pe.value(m.obj)}")
for v in m.component_data_objects(ctype=pe.Var):
    if "exp" in v.name and pe.value(v) >= .001:
        print(v.name, pe.value(v))
