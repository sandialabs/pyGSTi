
import pytest
from pygsti.tools.graphcoloring import switchboard_find_edge_coloring, check_valid_edge_coloring, find_edge_coloring
import numpy as np


# ALGORITHMS = ["new_bipartite", "assadi", "vizing", "sinnamon", "moser_tardos", "misra-gries"]
# ALGORITHMS = ["sinnamon", "misra_gries"]
ALGORITHMS = ["new_bipartite", "vizing", "sinnamon", "misra_gries"]

class TestGraphColoring(object):

    def test_find_edge_coloring_cycle_graph(self):
        # Define a cycle graph with 10 vertices
        num_vertices = 10
        vertices = list(range(num_vertices))
        edges = []
        neighbors = {i: [] for i in vertices}

        for i in range(num_vertices):
            u, v = i, (i + 1) % num_vertices
            edges.append((u, v))
            neighbors[u].append(v)
            neighbors[v].append(u)
        
        # Max degree for a cycle graph is 2
        deg = 2

        color_patches = find_edge_coloring(deg, vertices, edges, neighbors)

        # 1. Verify that each edge is colored exactly once
        # Collect all colored edges from color_patches
        colored_edges_from_patches = []
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                colored_edges_from_patches.append(tuple(sorted((u, v))))

        # Original unique edges (non-symmetric)
        original_unique_edges = set()
        for u, v in edges:
            original_unique_edges.add(tuple(sorted((u, v))))
        
        assert len(colored_edges_from_patches) == len(original_unique_edges), "Not all edges were colored or some were colored multiple times."
        assert len(set(colored_edges_from_patches)) == len(original_unique_edges), "Some edges were colored multiple times."

        # 2. Verify that no two adjacent edges have the same color
        # This means, for any vertex, all edges incident to it must have different colors.
        
        # Reconstruct edge_colors for easy lookup
        edge_colors = {}
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                edge_colors[tuple(sorted((u, v)))] = color
        
        for vertex in vertices:
            incident_edges = []
            for neighbor in neighbors[vertex]:
                incident_edges.append(tuple(sorted((vertex, neighbor))))
            
            # Get colors of incident edges
            incident_edge_colors = []
            for edge in incident_edges:
                # Some edges might not be present in edge_colors if they were implicitly handled
                # by symmetric pairs. We need to handle this.
                if edge in edge_colors:
                    incident_edge_colors.append(edge_colors[edge])
            
            # Check for duplicate colors among incident edges
            assert len(incident_edge_colors) == len(set(incident_edge_colors)), f"Vertex {vertex} has adjacent edges with the same color."

        # 3. Use the internal check_valid_edge_coloring for a quick verification
        check_valid_edge_coloring(color_patches)

        print(f"Edge coloring for cycle graph with {num_vertices} vertices passed all checks.")

    def test_find_edge_coloring_path_graph(self):
        # Define a path graph with 10 vertices
        num_vertices = 10
        vertices = list(range(num_vertices))
        edges = []
        neighbors = {i: [] for i in vertices}

        for i in range(num_vertices - 1):
            u, v = i, i + 1
            edges.append((u, v))
            neighbors[u].append(v)
            neighbors[v].append(u)

        # Max degree for a path graph is 2 (for internal vertices)
        # End vertices have degree 1
        deg = 2

        color_patches = find_edge_coloring(deg, vertices, edges, neighbors)

        # 1. Verify that each edge is colored exactly once
        colored_edges_from_patches = []
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                colored_edges_from_patches.append(tuple(sorted((u, v))))

        original_unique_edges = set()
        for u, v in edges:
            original_unique_edges.add(tuple(sorted((u, v))))

        assert len(colored_edges_from_patches) == len(original_unique_edges), "Not all edges were colored or some were colored multiple times."
        assert len(set(colored_edges_from_patches)) == len(original_unique_edges), "Some edges were colored multiple times."

        # 2. Verify that no two adjacent edges have the same color
        edge_colors = {}
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                edge_colors[tuple(sorted((u, v)))] = color

        for vertex in vertices:
            incident_edges = []
            for neighbor in neighbors[vertex]:
                incident_edges.append(tuple(sorted((vertex, neighbor))))

            incident_edge_colors = []
            for edge in incident_edges:
                if edge in edge_colors:
                    incident_edge_colors.append(edge_colors[edge])

            print(incident_edge_colors)
            assert len(incident_edge_colors) == len(set(incident_edge_colors)), f"Vertex {vertex} has adjacent edges with the same color."

        # 3. Use the internal check_valid_edge_coloring for a quick verification
        check_valid_edge_coloring(color_patches)

        print(f"Edge coloring for path graph with {num_vertices} vertices passed all checks.")

    def test_find_edge_coloring_high_degree_graph(self):
        # Define a graph with 10 vertices and a max degree of 5
        num_vertices = 10
        vertices = list(range(num_vertices))
        edges = []
        neighbors = {i: [] for i in vertices}

        # Connect vertex 0 to 5 other vertices to ensure a high degree
        high_degree_vertex = 0
        for i in range(1, 6):
            u, v = high_degree_vertex, i
            edges.append((u, v))
            neighbors[u].append(v)
            neighbors[v].append(u)
        
        # Add some more edges to other vertices
        edges_to_add = [
            (1, 6), (6, 1),
            (2, 7), (7, 2),
            (3, 8), (8, 3),
            (4, 9), (9, 4),
        ]
        for u, v in edges_to_add:
            if (u, v) not in edges and (v, u) not in edges: # Avoid duplicate edges
                edges.append((u, v))
                neighbors[u].append(v)
                neighbors[v].append(u)

        # Calculate the maximum degree dynamically
        deg = max(len(neighbors[v]) for v in vertices)

        color_patches = find_edge_coloring(deg, vertices, edges, neighbors)

        # 1. Verify that each edge is colored exactly once
        colored_edges_from_patches = []
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                colored_edges_from_patches.append(tuple(sorted((u, v))))

        original_unique_edges = set()
        for u, v in edges:
            original_unique_edges.add(tuple(sorted((u, v))))

        assert len(colored_edges_from_patches) == len(original_unique_edges), "Not all edges were colored or some were colored multiple times."
        assert len(set(colored_edges_from_patches)) == len(original_unique_edges), "Some edges were colored multiple times."

        # 2. Verify that no two adjacent edges have the same color
        edge_colors = {}
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                edge_colors[tuple(sorted((u, v)))] = color

        for vertex in vertices:
            incident_edges = []
            for neighbor in neighbors[vertex]:
                incident_edges.append(tuple(sorted((vertex, neighbor))))

            incident_edge_colors = []
            for edge in incident_edges:
                if edge in edge_colors:
                    incident_edge_colors.append(edge_colors[edge])

            assert len(incident_edge_colors) == len(set(incident_edge_colors)), f"Vertex {vertex} has adjacent edges with the same color. \n {incident_edge_colors} \n {set(incident_edge_colors)}"

        # 3. Use the internal check_valid_edge_coloring for a quick verification
        check_valid_edge_coloring(color_patches)

        print(f"Edge coloring for high degree graph with {num_vertices} vertices and max degree {deg} passed all checks.")

