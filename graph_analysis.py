import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
import random
import os
import csv

from collections import deque, Counter
from typing import Union
from scipy import stats
from matplotlib.animation import FuncAnimation

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the graph analysis program.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - graph_file: Positional path to input .gml file (optional)
            - input: Legacy --input flag for backward compatibility
            - output: Path to output .gml file (optional)
            - create_random_graph: Tuple of (n, c) for graph generation (optional)
            - multi_BFS: List of node IDs for BFS roots (optional)
            - analyze: Boolean flag to perform structural analysis
            - plot: Plot mode string 'C', 'N', 'P', or 'BFS' (default BFS when flag given alone)
            - components: Integer n for Girvan-Newman community partitioning
            - robustness_check: Integer k — remove k edges before partitioning (with --components)
                                  or run 10-round robustness simulation (standalone)
            - split_output_dir: Directory path to export each component as a .gml file
            - verify_homophily: Node attribute name to test (default 'color')
            - verify_balanced_graph: Boolean flag to check signed-graph balance
            - simulate_failures: Integer k edges to remove for single failure simulation
            - temporal_simulation: Path to CSV file for temporal edge animation
    """
    parser = argparse.ArgumentParser(
        description='Graph analysis tool for social and large-scale networks.',
        usage='python %(prog)s graph_file.gml [OPTIONS]'
    )

    # Positional argument (optional) — new canonical syntax
    parser.add_argument('graph_file',
                        type=str,
                        nargs='?',
                        metavar='graph_file.gml',
                        help='Path to the input .gml file.')

    # Legacy --input flag kept for backward compatibility
    parser.add_argument('--input',
                        type=str,
                        metavar='graph_file.gml',
                        help='(Legacy) Path to the input .gml file.')

    parser.add_argument('--output',
                        type=str,
                        metavar='output_file.gml',
                        help='Save graph to GML file.')

    parser.add_argument('--create_random_graph',
                        nargs=2,
                        type=float,
                        metavar=('n', 'c'),
                        help='Create random Erdos-Renyi graph with n nodes and constant c.')

    parser.add_argument('--multi_BFS',
                        nargs='+',
                        metavar='node_id',
                        type=str,
                        help='Perform BFS from one or more node IDs.')

    parser.add_argument('--analyze',
                        action='store_true',
                        help='Perform structural analysis on the graph.')

    # --plot changed from boolean to mode selector
    # nargs='?' means: absent→None, bare --plot→'BFS', --plot X→'X'
    parser.add_argument('--plot',
                        nargs='?',
                        const='BFS',
                        default=None,
                        metavar='MODE',
                        help='Visualization mode: C (clustering coefficient), '
                             'N (neighborhood overlap), P (signed/colored attributes), '
                             'BFS (default — BFS tree view).')

    # --- New Assignment 2 arguments ---

    parser.add_argument('--components',
                        type=int,
                        metavar='n',
                        help='Partition graph into n communities using Girvan-Newman.')

    parser.add_argument('--robustness_check',
                        type=int,
                        metavar='k',
                        help='With --components: remove k random edges before partitioning. '
                             'Standalone: run 10 rounds of k-edge-failure simulations.')

    parser.add_argument('--split_output_dir',
                        type=str,
                        metavar='dir',
                        help='Export each partition component to a separate .gml file '
                             'in this directory (requires --components).')

    parser.add_argument('--verify_homophily',
                        nargs='?',
                        const='color',
                        metavar='attribute',
                        help='Test homophily on the given node attribute (default: color).')

    parser.add_argument('--verify_balanced_graph',
                        action='store_true',
                        help='Check if signed graph (edge attribute sign=+1/-1) is balanced.')

    parser.add_argument('--simulate_failures',
                        type=int,
                        metavar='k',
                        help='Remove k random edges and report structural impact.')

    parser.add_argument('--temporal_simulation',
                        type=str,
                        metavar='file.csv',
                        help='Animate graph evolution from CSV (source,target,timestamp,action).')

    return parser.parse_args()


def display_graph(graph:nx.Graph) ->None:
    """
    Display a network graph visualization with descriptive title and statistics.
    
    Creates a matplotlib figure showing the graph with spring layout,
    labeled nodes, and a title containing node/edge counts.
    
    Args:
        graph: NetworkX graph to visualize
    
    Returns:
        None (displays plot to screen)
    """
    fig, ax = plt.subplots(figsize=(10,8))
    pos = nx.spring_layout(graph, seed=42)
    
    #add descriptive title with graph  statistics
    title = f"Generated Erdős–Rényi Random Graph\n{len(graph.nodes)} nodes, {len(graph.edges)} edges"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    nx.draw(graph,
            pos=pos,
            ax=ax,
            with_labels=True, 
            node_color="skyblue", 
            node_size=500,
            edge_color="gray",
            font_size=12,
            font_weight="bold")
    
    ax.axis("off")
    plt.show()


def generate_random_graph(n:int, c:float) -> nx.Graph:
    """
    Generate an Erdős–Rényi random graph using the G(n,p) model.
    
    Creates a graph with n nodes where each possible edge is included
    independently with probability p = (c * ln(n)) / n.
    
    Args:
        n: Number of nodes in the graph
        c: Constant multiplier for edge probability calculation
    
    Returns:
        NetworkX Graph with n nodes labeled as strings "0" to "n-1"
        and edges created probabilistically
    """
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(n)])
    p = c * math.log(n) / n
    # print(f"random graph prop: {p}")

    for i in range(n):
        for j in range(i+1, n): 
            if random.random()< p: 
                graph.add_edge(str(i), str(j))

    return graph


def BFS(graph: nx.Graph, node:str, return_visited:bool=False):
    """
    Perform Breadth-First Search from a starting node and create a BFS tree.
    
    Traverses the graph level by level from the starting node, creating a
    tree structure containing all reachable nodes and the edges that form
    shortest paths to them.
    
    Args:
        graph: The graph to traverse
        node: The starting node ID for BFS traversal
        return_visited: If True, also returns the set of visited nodes
    
    Returns:
        If return_visited is False: BFS tree as NetworkX Graph
        If return_visited is True: Tuple of (BFS tree, set of visited nodes)
    """

    visited = set()
    queue = deque([node])
    BFS_graph = nx.Graph()

    while queue: 
        curr_node = queue.popleft()
        visited.add(curr_node)
        BFS_graph.add_node(curr_node)
        neighbords = graph[curr_node]

        #process neighbords
        for neighbord in neighbords:
            if neighbord not in visited:
                queue.append(neighbord)
                visited.add(neighbord)
                BFS_graph.add_node(neighbord)
                BFS_graph.add_edge(curr_node, neighbord)

    if return_visited:
        return BFS_graph, visited
    else:
        return BFS_graph


def has_cycle(graph: nx.Graph) -> bool:
    """
    Determine whether the graph contains any cycles using depth-first search.
    
    A cycle is a path that starts and ends at the same node without repeating
    any edges or nodes (except the start/end node). Uses iterative DFS with
    parent tracking to detect back edges.
    
    Args:
        graph: The graph to check for cycles
    
    Returns:
        True if the graph contains at least one cycle, False otherwise
    """
    visited = set()

    for start in graph.nodes:
        if start in visited:
            continue

        # Stack holds (node, parent) pairs
        stack = [(start, None)]

        while stack:
            node, parent = stack.pop()

            if node in visited:
                continue

            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append((neighbor, node))
                elif neighbor != parent:
                    # Found visited node that's not parent = cycle!
                    return True

    return False


def get_isolated_nodes(graph:nx.Graph) -> list[str]:
    """
    Find all isolated nodes (nodes with no connections) in the graph.
    
    An isolated node is a node with degree 0, meaning it has no edges
    connecting it to any other node in the graph.
    
    Args:
        graph: The graph to search for isolated nodes
    
    Returns:
        List of node IDs that are isolated (empty list if none exist)
    """
    iso_nodes = []
    for node in graph.nodes:
        if len(graph[node]) == 0:
            iso_nodes.append(node)
    
    return iso_nodes


def find_all_cycles(graph: nx.Graph) -> list[list[str]]:
    """
    Find all simple cycles in the graph.
    
    Args:
        graph: The input graph
    
    Returns:
        List of cycles, where each cycle is a list of nodes
    """
    try:
        cycles = list(nx.simple_cycles(graph.to_directed()))
        # Remove duplicatee cycles (reverse direction)
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            cycle_tuple = tuple(sorted(cycle))
            if cycle_tuple not in seen:
                seen.add(cycle_tuple)
                unique_cycles.append(cycle)
        return unique_cycles
    except:
        return []


def tree_layout(graph:nx.Graph, root:str) -> dict:
      """
      Calculate node positions for tree visualization based on BFS levels.
      
      Arranges nodes in a hierarchical layout where nodes at the same
      distance from the root appear on the same horizontal level.
      
      Args:
          graph: The tree graph to layout
          root: The root node ID to start from
      
      Returns:
          Dictionary mapping node IDs to (x, y) coordinate tuples
      """
      levels = {root: 0}
      queue = deque([root])

      while queue:
          node = queue.popleft()
          for neighbor in graph[node]:
              if neighbor not in levels:
                  levels[neighbor] = levels[node] + 1
                  queue.append(neighbor)

      # Group nodes by level
      by_level = {}
      for node, level in levels.items():
          by_level.setdefault(level, []).append(node)

      # Assign positions
      pos = {}
      for level, nodes in by_level.items():
          for i, node in enumerate(nodes):
              x = (i + 1) / (len(nodes) + 1) 
              y = -level                       
              pos[node] = (x, y)

      return pos


def display_multiple_graph(bfs_graphs:list[nx.Graph], start_nodes:str) -> None:
    """
    Display multiple BFS trees side-by-side with titles and annotations.
    
    Creates a figure with subplots showing each BFS tree in hierarchical
    layout, with root nodes highlighted and statistics displayed.
    
    Args:
        bfs_graphs: List of BFS tree graphs to display
        start_nodes: List of root node IDs corresponding to each BFS tree
    
    Returns:
        None (displays plot to screen)
    """
    n = len(bfs_graphs)
    
    # Handle single subplot case
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    
    #Add overall figure title
    fig.suptitle('BFS Trees from Multiple Root Nodes', fontsize=16, fontweight='bold', y=0.98)
    
    for graph, ax, root in zip(bfs_graphs, axes, start_nodes):
        pos = tree_layout(graph, root=root)
        nx.draw(
            graph, 
            ax=ax,
            pos=pos,
            with_labels=True,
            node_color="lightgreen",
            node_size=500,
            edge_color="gray",
            font_size=12,
            font_weight="bold"
        )
        
        # Highlight root node
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=[root],
            node_color="orange",
            node_size=700,
            ax=ax
        )
        
        #add title for each subplot
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        ax.set_title(f"BFS from Node '{root}'\n{num_nodes} nodes reached, {num_edges} edges", 
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Adjust layout to prevent overlap with suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def count_components(graph:nx.Graph) -> int:
    """
    Count the number of connected components in the graph.
    
    A connected component is a maximal subgraph where every pair of nodes
    is connected by some path. Uses BFS to traverse and identify components.
    
    Args:
        graph: The graph to analyze
    
    Returns:
        Integer count of connected components
    """ 
    visited = set()
    count = 0

    for node in graph.nodes:
        #if the node is already visited, that's mean the cluster is already counted.
        if node in visited:
            continue
        
        count += 1

        _, temp_visited = BFS(graph, node, return_visited=True)
        visited.update(temp_visited)

    return count

def density(graph:nx.Graph) -> float:
    """
    Calculate the density of the graph.
    
    Graph density measures the ratio of actual edges to the maximum possible
    edges in an undirected graph: density = 2*E / (N*(N-1))
    where E is the number of edges and N is the number of nodes.
    
    Args:
        graph: The graph to analyze
    
    Returns:
        Float value between 0 (no edges) and 1 (complete graph)
    """
    n = len(graph.nodes)
    edges = len(graph.edges)

    return 2 * edges / (n * (n-1))


def average_shortest_path(graph:nx.Graph) -> dict[str, float]:
    """
    Compute average shortest path length for each connected component.
    
    For graphs with multiple components, calculates the average separately
    for each component (path length only defined within components).
    
    Args:
        graph: The graph to analyze
    
    Returns:
        Dictionary mapping component node lists (as strings) to their
        average shortest path length
    """
    res = {}

    for C in (graph.subgraph(c).copy() for c in nx.connected_components(graph)):
        key = "-".join(list(C.nodes))
        res[key] = nx.average_shortest_path_length(C)
    
    return res


def analyze_graph(graph:nx.Graph):
    """
    Perform comprehensive structural analysis on the graph and print results.
    
    Analyzes and displays:
    - Number of connected components
    - Presence of cycles
    - List of isolated nodes
    - Graph density (formatted to 3 decimal places)
    - Average shortest path length per component (formatted to 3 decimal places)
    
    Args:
        graph: The graph to analyze
    
    Returns:
        None (prints results to console)
    """
    print(f"Components count: {count_components(graph)}")
    print(f"Has cycle: {has_cycle(graph)}")
    print(f"Isolated nodes: {get_isolated_nodes(graph)}")
    print(f"Density: {density(graph):.3f}")
    
    avg_paths = average_shortest_path(graph)
    print("Average shortest path:", end=" ")
    if avg_paths:
        formatted_paths = {k: f"{v:.3f}" for k, v in avg_paths.items()}
        print(formatted_paths)
    else:
        print("N/A")

    return


def extract_bfs_info_from_graph(graph: nx.Graph) -> tuple[list[str], list[nx.Graph]]:
    """
    Extract BFS root nodes and reconstruct BFS trees from saved graph attributes.
    
    Args:
        graph: Graph with saved BFS attributes
    
    Returns:
        Tuple of (root_nodes, BFS_graphs)
    """
    root_nodes = []
    BFS_graphs = []
    
    # Check if graph has is_root attribute
    if not graph.nodes:
        return root_nodes, BFS_graphs
    
    # Get first node to check what attributes exist
    sample_node = list(graph.nodes)[0]
    node_attrs = graph.nodes[sample_node]
    
    # Find root nodes from is_root attribute
    if 'is_root' in node_attrs:
        root_nodes = [node for node, data in graph.nodes(data=True) if data.get('is_root', False)]
    
    # Reconstruct BFS trees from saved edge attributes
    for root in root_nodes:
        bfs_tree = nx.Graph()
        # Check if edge attributes exist for this root
        if graph.edges:
            sample_edge = list(graph.edges)[0]
            edge_attr_key = f'in_bfs_tree_{root}'
            
            if edge_attr_key in graph.edges[sample_edge]:
                # Add nodes and edges that belong to this BFS tree
                for u, v, data in graph.edges(data=True):
                    if data.get(edge_attr_key, False):
                        bfs_tree.add_edge(u, v)
                
                BFS_graphs.append(bfs_tree)
    
    return root_nodes, BFS_graphs


def add_graph_attributes(graph: nx.Graph, BFS_graphs: list[nx.Graph], root_nodes: list[str]):
    """
    Add comprehensive computed attributes to graph nodes and edges.
    
    Node attributes added:
    - component_id: Connected component identifier
    - is_isolated: Boolean flag for isolated nodes
    - is_root: Boolean flag for BFS root nodes
    - distance_from_{root}: Shortest path distance from each BFS root
    - parent_from_{root}: Parent node in BFS tree from each root
    - path_from_{root}: Full shortest path string from each root
    - in_cycle: Boolean flag indicating if node is part of any cycle
    
    Edge attributes added:
    - in_bfs_tree_{root}: Boolean flag for each BFS root
    - bfs_tree_count: Number of BFS trees this edge belongs to
    
    Args:
        graph: The graph to add attributes to
        BFS_graphs: List of BFS tree graphs
        root_nodes: List of BFS root node IDs
    
    Returns:
        The graph with added attributes
    """
    # add component ids
    component_map = {}
    for i, component in enumerate(nx.connected_components(graph)):
        for node in component:
            component_map[node] = i
    nx.set_node_attributes(graph, component_map, 'component_id')
    
    # add isolated node flags
    isolated_nodes = get_isolated_nodes(graph)
    is_isolated = {node: (node in isolated_nodes) for node in graph.nodes}
    nx.set_node_attributes(graph, is_isolated, 'is_isolated')
    
    # add root node flags
    is_root = {node: (node in root_nodes) for node in graph.nodes}
    nx.set_node_attributes(graph, is_root, 'is_root')
    
    # find cycles and mark nodes in cycles
    cycles = find_all_cycles(graph)
    nodes_in_cycles = set()
    for cycle in cycles:
        nodes_in_cycles.update(cycle)
    in_cycle = {node: (node in nodes_in_cycles) for node in graph.nodes}
    nx.set_node_attributes(graph, in_cycle, 'in_cycle')
    
    # add bfs-related attributes for each root
    for root, bfs_tree in zip(root_nodes, BFS_graphs):
        try:
            # calculate distances from root
            distances = nx.single_source_shortest_path_length(graph, root)
            nx.set_node_attributes(graph, distances, f'distance_from_{root}')
            
            # add parent nodes from bfs tree
            parents = {}
            bfs_tree_directed = nx.bfs_tree(graph, root)
            for node in bfs_tree_directed.nodes():
                if node != root:
                    predecessors = list(bfs_tree_directed.predecessors(node))
                    if predecessors:
                        parents[node] = predecessors[0]
            nx.set_node_attributes(graph, parents, f'parent_from_{root}')
            
            # add full path strings
            paths = nx.single_source_shortest_path(graph, root)
            path_strings = {node: '->'.join(path) for node, path in paths.items()}
            nx.set_node_attributes(graph, path_strings, f'path_from_{root}')
            
        except nx.NetworkXError:
            # skip if root is not in graph or unreachable nodes exist
            pass
    
    # add edge attributes
    # initialize edge attributes
    for edge in graph.edges():
        graph.edges[edge]['bfs_tree_count'] = 0
    
    # mark edges that are part of BFS trees
    for root, bfs_tree in zip(root_nodes, BFS_graphs):
        for edge in graph.edges():
            # check if edge (in either direction) is in the BFS tree
            u, v = edge
            is_in_tree = bfs_tree.has_edge(u, v) or bfs_tree.has_edge(v, u)
            graph.edges[edge][f'in_bfs_tree_{root}'] = is_in_tree
            
            if is_in_tree:
                graph.edges[edge]['bfs_tree_count'] += 1
    
    return graph


def partition_graph(graph: nx.Graph, n: int) -> list[nx.Graph]:
    """
    Partition the graph into n communities using the Girvan-Newman algorithm.

    Iteratively removes the edge with the highest betweenness centrality from
    a working copy until the graph has at least n connected components.

    Args:
        graph: The graph to partition (not mutated; a copy is used).
        n: Target number of communities/partitions.

    Returns:
        List of n subgraphs (nx.Graph), sorted by size descending.

    Raises:
        ValueError: If n < 1 or n > number of nodes in the graph.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n > len(graph.nodes):
        raise ValueError(f"n ({n}) cannot exceed the number of nodes ({len(graph.nodes)}).")

    working = graph.copy()
    current = count_components(working)

    if current >= n:
        # Already have enough components — return the n largest
        components = sorted(nx.connected_components(working), key=len, reverse=True)
        return [working.subgraph(c).copy() for c in components[:n]]

    print(f"Girvan-Newman partitioning: {current} → {n} components...")
    while current < n:
        if len(working.edges) == 0:
            print("Warning: No more edges to remove.")
            break

        # Remove edge with highest betweenness centrality
        betweenness = nx.edge_betweenness_centrality(working)
        edge_to_remove = max(betweenness, key=betweenness.get)
        working.remove_edge(*edge_to_remove)
        current = count_components(working)
        print(f"  Removed edge {edge_to_remove} → {current} component(s)")

    components = sorted(nx.connected_components(working), key=len, reverse=True)
    return [working.subgraph(c).copy() for c in components[:n]]


def export_components(components: list[nx.Graph], output_dir: str) -> None:
    """
    Export each component graph to a separate .gml file.

    Files are written as component_0.gml, component_1.gml, etc. (ordered
    by descending size). The directory is created if it does not exist.

    Args:
        components: List of component subgraphs to export.
        output_dir: Path to the output directory.

    Returns:
        None (writes files and prints confirmation).
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, comp in enumerate(components):
        filepath = os.path.join(output_dir, f"component_{i}.gml")
        nx.write_gml(comp, filepath)
        print(f"  Exported component {i} ({len(comp.nodes)} nodes, "
              f"{len(comp.edges)} edges) → {filepath}")


def plot_clustering(graph: nx.Graph) -> None:
    """
    Visualize the graph with node size proportional to clustering coefficient
    and node color encoding degree.

    Node size = 300 + CC * 1500 (larger = more locally clustered).
    Node color = degree, mapped through plt.cm.plasma colormap.

    Args:
        graph: The graph to visualize.

    Returns:
        None (displays plot to screen).
    """
    clustering = nx.clustering(graph)
    degrees = dict(graph.degree())

    node_sizes = [300 + clustering[n] * 1500 for n in graph.nodes]

    max_deg = max(degrees.values()) if degrees else 1
    node_color_vals = [degrees[n] / max_deg for n in graph.nodes]

    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 9))

    nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_color_vals,
        cmap=plt.cm.plasma,
        vmin=0, vmax=1
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_weight='bold')

    # Colorbar for degree
    sm = cm.ScalarMappable(cmap=plt.cm.plasma, norm=mcolors.Normalize(vmin=0, vmax=max_deg))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Degree')

    ax.set_title(
        'Clustering Coefficient Visualization\n'
        'Node size = clustering coefficient  |  Node color = degree',
        fontsize=13, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_neighborhood_overlap(graph: nx.Graph) -> None:
    """
    Visualize the graph with edge thickness proportional to neighborhood overlap
    and edge color representing the sum of endpoint degrees.

    Neighborhood overlap for edge (u,v):
        NO = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    where N(x) excludes the edge endpoints themselves.

    Edge width = 1 + NO * 8.
    Edge color = normalized sum of endpoint degrees (plt.cm.coolwarm).

    Args:
        graph: The graph to visualize.

    Returns:
        None (displays plot to screen).
    """
    edge_overlap = {}
    edge_deg_sum = {}

    for u, v in graph.edges():
        n_u = set(graph.neighbors(u)) - {v}
        n_v = set(graph.neighbors(v)) - {u}
        union_size = len(n_u | n_v)
        overlap = len(n_u & n_v) / union_size if union_size > 0 else 0.0
        edge_overlap[(u, v)] = overlap
        edge_deg_sum[(u, v)] = graph.degree(u) + graph.degree(v)

    edges = list(graph.edges())
    widths = [1 + edge_overlap[e] * 8 for e in edges]
    deg_sums = [edge_deg_sum[e] for e in edges]

    max_ds = max(deg_sums) if deg_sums else 1
    edge_colors = [ds / max_ds for ds in deg_sums]

    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 9))

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='lightblue',
                           node_size=500, edgecolors='black', linewidths=1)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edgelist=edges,
        width=widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.coolwarm,
        edge_vmin=0, edge_vmax=1
    )

    # Colorbar for degree sum
    sm = cm.ScalarMappable(cmap=plt.cm.coolwarm,
                           norm=mcolors.Normalize(vmin=0, vmax=max_ds))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Sum of endpoint degrees')

    ax.set_title(
        'Neighborhood Overlap Visualization\n'
        'Edge thickness = neighborhood overlap  |  Edge color = sum of endpoint degrees',
        fontsize=13, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_signed_graph(graph: nx.Graph) -> None:
    """
    Visualize a signed graph using node 'color' attribute and edge 'sign' attribute.

    Node color is read from the 'color' node attribute (defaults to 'lightblue').
    Positive edges (sign=+1) are drawn in green; negative edges (sign=-1) in red.
    Unsigned edges are drawn in gray.

    Args:
        graph: The graph to visualize. Expected node attribute: 'color' (str).
               Expected edge attribute: 'sign' (int: +1 or -1).

    Returns:
        None (displays plot to screen).
    """
    node_colors = [graph.nodes[n].get('color', 'lightblue') for n in graph.nodes]

    pos_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('sign', 0) > 0]
    neg_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('sign', 0) < 0]
    neu_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('sign', 0) == 0]

    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 9))

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                           node_size=600, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10, font_weight='bold')

    if pos_edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pos_edges,
                               edge_color='green', width=2.5, label='+1 (positive)')
    if neg_edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=neg_edges,
                               edge_color='red', width=2.5, style='dashed', label='-1 (negative)')
    if neu_edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=neu_edges,
                               edge_color='gray', width=1, label='unsigned')

    # Legend
    legend_elements = []
    if pos_edges:
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], color='green', linewidth=2.5, label='Positive edge (+1)')
        )
    if neg_edges:
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], color='red', linewidth=2.5,
                                    linestyle='dashed', label='Negative edge (-1)')
        )
    if neu_edges:
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], color='gray', linewidth=1, label='Unsigned edge')
        )

    # Get unique node color groups for legend
    color_groups = {}
    for node in graph.nodes:
        c = graph.nodes[node].get('color', 'lightblue')
        color_groups.setdefault(c, []).append(node)
    for c, nodes in color_groups.items():
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=c, markersize=12,
                                    markeredgecolor='black', markeredgewidth=1,
                                    label=f"color='{c}' ({len(nodes)} nodes)", linestyle='')
        )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('Signed Graph — Node Colors and Edge Signs', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def verify_homophily(graph: nx.Graph, attribute: str = 'color') -> None:
    """
    Statistical test (one-sample t-test) for homophily on a node attribute.

    For each node with degree > 0, computes the fraction of its neighbors
    that share its own attribute value (f_n). The global base rate (p_n) for
    that attribute value is the fraction of all nodes with the same value.
    The difference (f_n - p_n) is tested against zero using scipy.stats.ttest_1samp
    with alternative='greater'. A significantly positive mean indicates homophily.

    Args:
        graph: The graph to analyze.
        attribute: Node attribute to test (default 'color').

    Returns:
        None (prints t-statistic, p-value, and plain-English interpretation).
    """
    print(f"\n=== Homophily Test (attribute: '{attribute}') ===")

    # Collect attribute values
    attr_values = {n: graph.nodes[n].get(attribute) for n in graph.nodes}
    valid_nodes = [n for n, v in attr_values.items() if v is not None]

    if not valid_nodes:
        print(f"Error: No nodes have the attribute '{attribute}'.")
        return

    # Global base rate for each attribute value
    value_counts = Counter(attr_values[n] for n in valid_nodes)
    total_nodes = len(valid_nodes)
    base_rates = {val: count / total_nodes for val, count in value_counts.items()}

    print(f"Attribute distribution: {dict(value_counts)}")
    print(f"Base rates: { {k: f'{v:.3f}' for k,v in base_rates.items()} }")

    # Compute per-node difference: (fraction of same-attr neighbors) - (base rate)
    differences = []
    for n in valid_nodes:
        if graph.degree(n) == 0:
            continue
        my_val = attr_values[n]
        neighbors = list(graph.neighbors(n))
        same_count = sum(1 for nb in neighbors if attr_values.get(nb) == my_val)
        f_n = same_count / len(neighbors)
        p_n = base_rates[my_val]
        differences.append(f_n - p_n)

    if len(differences) < 2:
        print("Not enough connected nodes to perform statistical test.")
        return

    t_stat, p_value = stats.ttest_1samp(differences, 0, alternative='greater')
    mean_diff = sum(differences) / len(differences)

    print(f"Mean excess same-attribute neighbor fraction: {mean_diff:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (one-tailed): {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"Result: HOMOPHILY DETECTED (p={p_value:.4f} < {alpha}) — "
              f"nodes connect significantly more to same-attribute neighbors.")
    else:
        print(f"Result: No significant homophily (p={p_value:.4f} >= {alpha}).")


def verify_balanced_graph(graph: nx.Graph) -> bool:
    """
    Check if a signed graph is structurally balanced using BFS 2-coloring.

    A signed graph is balanced if and only if nodes can be 2-partitioned such
    that all positive edges (sign=+1) are within partitions and all negative
    edges (sign=-1) are between partitions. This is checked via BFS:
    assign color 0 to the start node, propagate same color across positive
    edges and opposite color across negative edges. A contradiction means
    the graph is not balanced.

    Edges without a 'sign' attribute are treated as positive (+1).

    Args:
        graph: The signed graph to check.

    Returns:
        True if balanced, False otherwise. Also prints result and partitions.
    """
    print("\n=== Signed Graph Balance Check ===")

    coloring = {}  # node -> 0 or 1
    balanced = True

    for start in graph.nodes:
        if start in coloring:
            continue

        coloring[start] = 0
        queue = deque([start])

        while queue and balanced:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                sign = graph.edges[node, neighbor].get('sign', 1)
                # +1 edge → same color; -1 edge → opposite color
                expected_color = coloring[node] if sign > 0 else 1 - coloring[node]

                if neighbor not in coloring:
                    coloring[neighbor] = expected_color
                    queue.append(neighbor)
                elif coloring[neighbor] != expected_color:
                    balanced = False
                    break

    if balanced:
        group_0 = [n for n, c in coloring.items() if c == 0]
        group_1 = [n for n, c in coloring.items() if c == 1]
        print(f"Result: BALANCED")
        print(f"  Group 0 (friends): {group_0}")
        print(f"  Group 1 (enemies): {group_1}")
    else:
        print("Result: NOT BALANCED — contradiction found in signed cycle.")

    return balanced


def simulate_failures(graph: nx.Graph, k: int) -> None:
    """
    Simulate k random edge failures and report the structural impact.

    Removes k edges chosen uniformly at random from a copy of the graph
    and measures changes in: number of connected components, average
    shortest path length per component, and betweenness centrality.

    Args:
        graph: The original graph (not mutated).
        k: Number of edges to remove.

    Returns:
        None (prints before/after comparison to console).
    """
    print(f"\n=== Simulating {k} Random Edge Failure(s) ===")

    all_edges = list(graph.edges())
    if k >= len(all_edges):
        print(f"Error: k={k} must be less than the number of edges ({len(all_edges)}).")
        return

    # Baseline metrics
    before_comp = count_components(graph)
    before_asp = average_shortest_path(graph)
    before_bc = nx.betweenness_centrality(graph)

    # Remove k random edges
    working = graph.copy()
    removed = random.sample(all_edges, k)
    working.remove_edges_from(removed)

    # Post-failure metrics
    after_comp = count_components(working)
    after_asp = average_shortest_path(working)
    after_bc = nx.betweenness_centrality(working)

    print(f"Removed edges: {removed}")
    print(f"\nConnected components: {before_comp} → {after_comp} "
          f"(Δ = {after_comp - before_comp:+d})")

    # Average shortest path comparison (compare largest component)
    def largest_component_asp(asp_dict):
        if not asp_dict:
            return None
        # The key for the largest component is the longest key string
        return max(asp_dict.items(), key=lambda x: len(x[0].split('-')))[1]

    before_val = largest_component_asp(before_asp)
    after_val = largest_component_asp(after_asp)
    if before_val is not None and after_val is not None:
        print(f"Avg shortest path (largest component): "
              f"{before_val:.3f} → {after_val:.3f} (Δ = {after_val - before_val:+.3f})")
    else:
        print("Avg shortest path: N/A (graph may be disconnected)")

    # Top-5 betweenness centrality nodes
    top5_before = sorted(before_bc.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_after = sorted(after_bc.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop-5 betweenness centrality nodes:")
    print(f"  Before: {[(n, f'{v:.4f}') for n, v in top5_before]}")
    print(f"  After:  {[(n, f'{v:.4f}') for n, v in top5_after]}")


def robustness_check(graph: nx.Graph, k: int, rounds: int = 10) -> None:
    """
    Run multiple simulations of k random edge failures to assess robustness.

    Performs 'rounds' independent simulations, each removing k distinct random
    edges and measuring component structure. Reports aggregate statistics and
    whether the original community structure tends to persist.

    Args:
        graph: The original graph (not mutated).
        k: Number of edges removed per simulation round.
        rounds: Number of independent simulation rounds (default 10).

    Returns:
        None (prints aggregate robustness report to console).
    """
    print(f"\n=== Robustness Check: {rounds} rounds of {k} random edge failure(s) ===")

    all_edges = list(graph.edges())
    if k >= len(all_edges):
        print(f"Error: k={k} must be less than the number of edges ({len(all_edges)}).")
        return

    # Original community structure
    orig_components = list(nx.connected_components(graph))
    orig_comp_count = len(orig_components)

    comp_counts = []
    largest_sizes = []
    smallest_sizes = []
    cluster_persists = 0

    for _ in range(rounds):
        working = graph.copy()
        removed = random.sample(list(working.edges()), k)
        working.remove_edges_from(removed)

        new_components = list(nx.connected_components(working))
        comp_counts.append(len(new_components))

        sizes = sorted([len(c) for c in new_components], reverse=True)
        largest_sizes.append(sizes[0])
        smallest_sizes.append(sizes[-1])

        # Check if original clusters persist (each orig component stays together)
        persists = True
        for orig_c in orig_components:
            # Check all nodes of this original component are still in the same new component
            found = False
            for new_c in new_components:
                if orig_c.issubset(new_c):
                    found = True
                    break
            if not found:
                persists = False
                break
        if persists:
            cluster_persists += 1

    import statistics
    mean_comp = statistics.mean(comp_counts)
    std_comp = statistics.stdev(comp_counts) if len(comp_counts) > 1 else 0.0

    print(f"Original graph: {orig_comp_count} component(s), {len(graph.nodes)} nodes, "
          f"{len(all_edges)} edges")
    print(f"\nOver {rounds} rounds of {k} edge removal(s):")
    print(f"  Components — mean: {mean_comp:.2f}, std: {std_comp:.2f}, "
          f"min: {min(comp_counts)}, max: {max(comp_counts)}")
    print(f"  Largest component size — "
          f"mean: {statistics.mean(largest_sizes):.1f}, "
          f"max: {max(largest_sizes)}, min: {min(largest_sizes)}")
    print(f"  Smallest component size — "
          f"mean: {statistics.mean(smallest_sizes):.1f}, "
          f"max: {max(smallest_sizes)}, min: {min(smallest_sizes)}")
    print(f"  Original clusters persist: {cluster_persists}/{rounds} rounds "
          f"({100 * cluster_persists / rounds:.0f}%)")


def temporal_simulation(csv_file: str) -> None:
    """
    Animate graph evolution from a temporal edge log CSV file.

    Reads a CSV with columns (source, target, timestamp, action) where
    action is 'add' or 'remove'. Sorts by timestamp, groups events into
    frames, and uses FuncAnimation to display the graph state at each step.

    Args:
        csv_file: Path to CSV file with columns:
                  source (str), target (str), timestamp (numeric), action (str).

    Returns:
        None (displays animation in a matplotlib window).
    """
    print(f"\n=== Temporal Simulation from '{csv_file}' ===")

    try:
        events = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append({
                    'source': row['source'],
                    'target': row['target'],
                    'timestamp': float(row['timestamp']),
                    'action': row['action'].strip().lower()
                })
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return
    except KeyError as e:
        print(f"Error: Missing expected CSV column: {e}. "
              f"Expected columns: source, target, timestamp, action.")
        return

    if not events:
        print("Error: CSV file is empty.")
        return

    # Sort events by timestamp and group into frames
    events.sort(key=lambda x: x['timestamp'])
    frames_dict = {}
    for ev in events:
        ts = ev['timestamp']
        frames_dict.setdefault(ts, []).append(ev)
    frame_list = sorted(frames_dict.items())  # list of (timestamp, [events])

    anim_graph = nx.Graph()
    # Pre-seed all nodes that will ever appear so layout is stable
    all_nodes = set()
    for ev in events:
        all_nodes.add(ev['source'])
        all_nodes.add(ev['target'])
    anim_graph.add_nodes_from(all_nodes)

    pos = nx.spring_layout(anim_graph, seed=42)  # fixed layout throughout animation

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Temporal Graph Evolution', fontsize=14, fontweight='bold')

    def update(frame_idx):
        ax.clear()
        ts, ev_list = frame_list[frame_idx]
        for ev in ev_list:
            src, tgt, action = ev['source'], ev['target'], ev['action']
            if action == 'add':
                anim_graph.add_edge(src, tgt)
            elif action == 'remove':
                if anim_graph.has_edge(src, tgt):
                    anim_graph.remove_edge(src, tgt)

        nx.draw_networkx(
            anim_graph, pos=pos, ax=ax,
            with_labels=True,
            node_color='skyblue', node_size=500,
            edge_color='steelblue', width=1.5,
            font_size=9, font_weight='bold'
        )
        ax.set_title(
            f"t = {ts:.0f}  |  "
            f"nodes: {len(anim_graph.nodes)}  |  "
            f"edges: {len(anim_graph.edges)}",
            fontsize=11
        )
        ax.axis('off')

    anim = FuncAnimation(fig, update, frames=len(frame_list),
                         interval=800, repeat=False)
    plt.tight_layout()
    plt.show()


def plot(graph:nx.Graph, root_nodes:list[str], BFS_graphs:list[nx.Graph]):
    """
    Visualize graph with highlighted BFS paths, isolated nodes, and comprehensive legend.
    
    Args:
        graph: The original graph to visualize
        root_nodes: List of BFS root node IDs
        BFS_graphs: List of BFS tree graphs
    """
    # create figure with larger size for better visibility
    plt.figure(figsize=(14, 10))
    
    # Use layout that spreads nodes well
    pos = nx.kamada_kawai_layout(graph)

    # Get distinct colors for each BFS tree
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(BFS_graphs))]
    iso_nodes = get_isolated_nodes(graph)
    node_size = 500
    
    # Calculate graph statistics for annotations
    num_components = count_components(graph)
    has_cycles = has_cycle(graph)
    graph_density = density(graph)

    # Draw base graph nodes (non-isolated)
    regular_nodes = [n for n in graph.nodes if n not in iso_nodes]
    nx.draw_networkx_nodes(
        graph, 
        pos, 
        nodelist=regular_nodes,
        node_size=node_size,
        node_color='lightblue',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Highlight isolated nodes in red
    if iso_nodes:
        nx.draw_networkx_nodes(
            graph, 
            pos=pos, 
            nodelist=iso_nodes,
            node_size=node_size,
            node_color='red',
            edgecolors='darkred',
            linewidths=2
        )
    
    # Highlight BFS root nodes
    if root_nodes:
        root_nodes_in_graph = [n for n in root_nodes if n in graph.nodes]
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=root_nodes_in_graph,
            node_size=700,
            node_color='gold',
            edgecolors='orange',
            linewidths=3
        )

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")
    
    # Draw base edges in light gray
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1)
    
    # Draw BFS tree edges with distinct colors
    legend_elements = []
    for i, (bfs_graph, root) in enumerate(zip(BFS_graphs, root_nodes)):
        edgelist = bfs_graph.edges
        nx.draw_networkx_edges(
            graph, 
            pos, 
            edgelist=edgelist, 
            edge_color=colors[i],
            width=2.5,
            alpha=0.7,
            connectionstyle=f"arc3,rad={0.1 * (i + 1)}",
            arrows=True,
            arrowsize=10,
            arrowstyle='->'
        )
        # Add to legend
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], color=colors[i], linewidth=2.5, 
                                   label=f"BFS from node '{root}'")
        )
    
    # Add legend elements for nodes
    legend_elements.extend([
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor='gold', markersize=12, 
                               markeredgecolor='orange', markeredgewidth=2,
                               label='BFS Root Nodes', linestyle=''),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor='lightblue', markersize=10,
                               markeredgecolor='black', markeredgewidth=1,
                               label='Regular Nodes', linestyle='')
    ])
    
    if iso_nodes:
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='red', markersize=10,
                                   markeredgecolor='darkred', markeredgewidth=1.5,
                                   label=f'Isolated Nodes ({len(iso_nodes)})', linestyle='')
        )
    
    # Create comprehensive title with graph statistics
    title_lines = [
        f"Graph Visualization with BFS Shortest Paths",
        f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)} | Components: {num_components} | "
        f"Density: {graph_density:.3f} | Cycles: {'Yes' if has_cycles else 'No'}"
    ]
    plt.title('\n'.join(title_lines), fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              framealpha=0.9, edgecolor='black')
    
    # Add text annotation with additional info
    if root_nodes:
        info_text = f"BFS performed from {len(root_nodes)} root node(s): {', '.join(root_nodes)}"
        plt.text(0.5, -0.05, info_text, transform=plt.gca().transAxes,
                ha='center', fontsize=9, style='italic', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()
    graph = nx.Graph()
    root_nodes = []
    BFS_graphs = []

    # ── GRAPH LOADING ─────────────────────────────────────────────────────────
    # Positional argument takes precedence over legacy --input flag
    input_path = args.graph_file or args.input

    if input_path:
        try:
            graph = nx.read_gml(input_path)
            loaded_roots, loaded_bfs_graphs = extract_bfs_info_from_graph(graph)
            if loaded_roots:
                root_nodes = loaded_roots
                BFS_graphs = loaded_bfs_graphs
                print(f"Loaded graph with BFS data from {len(root_nodes)} root(s): "
                      f"{', '.join(root_nodes)}")
        except FileNotFoundError:
            print(f"Error: File '{input_path}' not found. Please check the path and try again.")
            return 1
        except PermissionError:
            print(f"Error: Permission denied reading '{input_path}'.")
            return 1
        except nx.NetworkXError as e:
            print(f"Error: Invalid GML format in '{input_path}': {e}")
            return 1
        except Exception as err:
            print(f"Error reading GML file: {err}")
            return 1

    # ── RANDOM GRAPH GENERATION ───────────────────────────────────────────────
    if args.create_random_graph:
        try:
            n = int(args.create_random_graph[0])
            c = args.create_random_graph[1]
            if n <= 0:
                print("Error: Number of nodes must be positive.")
                return 1
            if c <= 0:
                print("Error: Constant c must be positive.")
                return 1
            graph = generate_random_graph(n, c)
            display_graph(graph)
        except ValueError as err:
            print(f"Error: Invalid parameters for random graph: {err}")
            return 1

    # ── BFS ───────────────────────────────────────────────────────────────────
    if args.multi_BFS:
        root_nodes = args.multi_BFS
        for param in root_nodes:
            if param not in graph.nodes:
                print(f"Error: Node '{param}' not found in graph.")
                print(f"Available nodes: {list(graph.nodes)}")
                return 1
            BFS_graphs.append(BFS(graph, param))
        display_multiple_graph(BFS_graphs, root_nodes)

    # ── OUTPUT (SAVE) ─────────────────────────────────────────────────────────
    if args.output:
        if len(graph) > 0:
            try:
                if BFS_graphs and root_nodes:
                    graph = add_graph_attributes(graph, BFS_graphs, root_nodes)
                    print(f"\nGraph saved to '{args.output}' with enriched attributes:")
                    print(f"  - Node attributes: component_id, is_isolated, is_root, in_cycle")
                    print(f"  - BFS attributes for {len(root_nodes)} root(s): distances, parents, paths")
                    print(f"  - Edge attributes: BFS tree membership, tree count")
                else:
                    print(f"\nGraph saved to '{args.output}'.")
                nx.write_gml(graph, args.output)
            except PermissionError:
                print(f"Error: Permission denied writing to '{args.output}'.")
                return 1
            except Exception as err:
                print(f"Error saving graph: {err}")
                return 1
        else:
            print("Error: No graph to save. Provide a graph file or use --create_random_graph.")
            return 1

    # ── ANALYSIS ──────────────────────────────────────────────────────────────
    if args.analyze:
        if len(graph) == 0:
            print("Error: No graph to analyze.")
            return 1
        analyze_graph(graph)

    # ── PLOT ──────────────────────────────────────────────────────────────────
    if args.plot is not None:
        if len(graph) == 0:
            print("Error: No graph to plot.")
            return 1
        valid_modes = {'C', 'N', 'P', 'BFS'}
        if args.plot not in valid_modes:
            print(f"Error: Invalid plot mode '{args.plot}'. Choose from: C, N, P, BFS.")
            return 1
        if args.plot == 'C':
            plot_clustering(graph)
        elif args.plot == 'N':
            plot_neighborhood_overlap(graph)
        elif args.plot == 'P':
            plot_signed_graph(graph)
        else:  # 'BFS' — original visualization
            plot(graph, root_nodes, BFS_graphs)

    # ── COMPONENTS (Girvan-Newman) ─────────────────────────────────────────────
    if args.components:
        if len(graph) == 0:
            print("Error: No graph loaded for partitioning.")
            return 1
        n_parts = args.components
        working_graph = graph.copy()

        # Optional: remove k edges before partitioning (robustness pre-processing)
        if args.robustness_check:
            k = args.robustness_check
            if k >= len(working_graph.edges):
                print(f"Error: k={k} must be less than the number of edges "
                      f"({len(working_graph.edges)}).")
                return 1
            edges_to_remove = random.sample(list(working_graph.edges()), k)
            working_graph.remove_edges_from(edges_to_remove)
            print(f"Removed {k} random edge(s) before partitioning: {edges_to_remove}")

        try:
            components = partition_graph(working_graph, n_parts)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        print(f"\nPartitioned into {len(components)} component(s):")
        for i, comp in enumerate(components):
            print(f"  Component {i}: {len(comp.nodes)} nodes, {len(comp.edges)} edges  "
                  f"— nodes: {list(comp.nodes)}")

        if args.split_output_dir:
            export_components(components, args.split_output_dir)

    # ── VERIFY HOMOPHILY ──────────────────────────────────────────────────────
    if args.verify_homophily is not None:
        if len(graph) == 0:
            print("Error: No graph loaded for homophily test.")
            return 1
        verify_homophily(graph, args.verify_homophily)

    # ── VERIFY BALANCED GRAPH ─────────────────────────────────────────────────
    if args.verify_balanced_graph:
        if len(graph) == 0:
            print("Error: No graph loaded for balance check.")
            return 1
        verify_balanced_graph(graph)

    # ── SIMULATE FAILURES ─────────────────────────────────────────────────────
    if args.simulate_failures is not None:
        if len(graph) == 0:
            print("Error: No graph loaded for failure simulation.")
            return 1
        if args.simulate_failures <= 0:
            print("Error: k for --simulate_failures must be a positive integer.")
            return 1
        simulate_failures(graph, args.simulate_failures)

    # ── ROBUSTNESS CHECK (standalone — only when --components is NOT used) ─────
    if args.robustness_check and not args.components:
        if len(graph) == 0:
            print("Error: No graph loaded for robustness check.")
            return 1
        if args.robustness_check <= 0:
            print("Error: k for --robustness_check must be a positive integer.")
            return 1
        robustness_check(graph, args.robustness_check)

    # ── TEMPORAL SIMULATION ───────────────────────────────────────────────────
    if args.temporal_simulation:
        temporal_simulation(args.temporal_simulation)

    return 0  # success

if __name__ == "__main__": 
    main()