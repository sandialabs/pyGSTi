# # NOTE: Ignore this function. I'm pretty sure it is not needed.
# def generate_edge_colorings(vertices: list, edges: list) -> list:
#     """
#     Generate a set of edge colorings for a graph until all edges are colored.

#     This function takes an edge set of a simple undirected graph and repeatedly 
#     applies the Misra & Gries edge coloring algorithm until every edge is 
#     contained in some edge coloring. It returns a dictionary mapping colors 
#     to the edges colored with that color.

#     Parameters:
#     vertices (list): A list of vertices in the graph.
#     edges (list): A list of edges represented as tuples (u, v) where u and v 
#                   are vertices in the graph.

#     Returns:
#     list: A list of edge colorings (dictionaries whose keys are colors and items are lists colored edges)
#     """
#     list_of_edge_colorings = []
#     uncolored_edges = set(edges)

#     while uncolored_edges:
#         # Determine which vertices are neighbors in a graph with only uncolored edges
#         # Could call find_neighbors here...
#         updated_neighbors = {v: [] for v in vertices}
#         for u, v in uncolored_edges:
#             updated_neighbors[u].append(v)

#         # Calculate the maximum degree of the graph
#         deg = max(len(updated_neighbors[v]) for v in vertices)

#         # Find an edge coloring
#         new_color_patches = find_edge_coloring(deg, vertices, list(uncolored_edges), updated_neighbors)

#         # Update color patches and remove newly colored edges from uncolored_edges
#         list_of_edge_colorings.append(new_color_patches)
#         for _, edge_list in new_color_patches.items():
#             uncolored_edges.difference_update(edge_list)
#             uncolored_edges.difference_update([(v,u) for u, v in edge_list]) # need to symmetrize

#     return list_of_edge_colorings

# # NOTE: This class is superfluous. Keeping it around in case I realize that it isn't - Daniel H.
# class CrosstalkFreeCombinedExperimentDesign(CombinedExperimentDesign, HasProcessorSpec):
#     def __init__(self, processor_spec, oneq_gstdesign, twoq_gstdesign, seed = None, interleave = False):
        
#         HasProcessorSpec.__init__(self, processor_spec)
        
#         randstate = np.random.RandomState(seed)
#         self.interleave = interleave
#         self.oneq_gstdesign = oneq_gstdesign
#         self.twoq_gstdesign = twoq_gstdesign
#         self.vertices = self.processor_spec.qubit_labels
#         self.edges = self.processor_spec.compute_2Q_connectivity().edges()
#         self.neighbors = find_neighbors(self.vertices, self.edges)
#         self.deg = max([len(self.neighbors[v]) for v in self.vertices])


#         # Generate the sub-experiment designs
#         self.edge_colorings = generate_edge_colorings(self.vertices, self.edges)
#         self.sub_designs = [CrosstalkFreeSubExperimentDesign(self.processor_spec, 
#                                                             self.oneq_gstdesign,
#                                                             self.twoq_gstdesign,
#                                                             edge_coloring,
#                                                             randstate) for edge_coloring in self.edge_colorings]
#         CombinedExperimentDesign.__init__(self, sub_designs = self.sub_designs, qubit_labels = self.vertices, interleave = self.interleave)
