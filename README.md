# 02_Assignment-Social-and-Large-Scale-Networks

A Python CLI for analyzing and visualizing complex graph data, including community detection, clustering metrics, homophily testing, signed-graph balance, edge failure simulation, robustness analysis, and temporal animation.

---

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` contains: `networkx`, `matplotlib`, `scipy`

---

## Usage

```
python graph_analysis.py graph_file.gml [OPTIONS]
```

The `.gml` file is a positional argument. Legacy `--input graph_file.gml` is also accepted.

### Options

| Flag | Description |
|------|-------------|
| `--plot [C\|N\|P\|BFS]` | Visualization mode (see below) |
| `--components n` | Partition into n communities (Girvan-Newman) |
| `--robustness_check k` | With `--components`: remove k edges before partitioning. Standalone: 10-round robustness simulation |
| `--split_output_dir dir` | Export each component to a `.gml` file in `dir` (requires `--components`) |
| `--verify_homophily [attr]` | t-test for homophily on node attribute (default: `color`) |
| `--verify_balanced_graph` | Check if signed graph (edge `sign` ±1) is balanced |
| `--simulate_failures k` | Remove k random edges, report structural impact |
| `--temporal_simulation file.csv` | Animate graph evolution from CSV (source, target, timestamp, action) |
| `--output out.gml` | Save graph with enriched attributes |
| `--analyze` | Print structural stats (components, density, shortest path, etc.) |
| `--multi_BFS node [node ...]` | BFS trees from one or more root nodes |
| `--create_random_graph n c` | Generate Erdős–Rényi random graph (n nodes, parameter c) |

### `--plot` modes

| Mode | Description |
|------|-------------|
| `C` | Node size = clustering coefficient; node color = degree |
| `N` | Edge thickness = neighborhood overlap; edge color = sum of endpoint degrees |
| `P` | Node color from `color` attribute; green/red edges from `sign` attribute (+1/-1) |
| `BFS` | Original BFS-path visualization (default when `--plot` given with no argument) |

---

## Sample Commands

```bash
# Clustering coefficient visualization
python graph_analysis.py sample_graph.gml --plot C

# Neighborhood overlap visualization
python graph_analysis.py sample_graph.gml --plot N

# Signed/colored attribute visualization
python graph_analysis.py sample_graph.gml --plot P

# Homophily test + balance check
python graph_analysis.py sample_graph.gml --verify_homophily --verify_balanced_graph

# Partition into 3 communities and export each component
python graph_analysis.py community_graph.gml --components 3 --split_output_dir ./parts

# Partition with 2 edges removed first (robustness pre-processing)
python graph_analysis.py community_graph.gml --components 3 --robustness_check 2

# Single edge failure simulation (remove 5 edges)
python graph_analysis.py community_graph.gml --simulate_failures 5

# Standalone robustness check (10 rounds of removing 3 edges)
python graph_analysis.py community_graph.gml --robustness_check 3

# Temporal animation
python graph_analysis.py sample_graph.gml --temporal_simulation temporal_edges.csv

# Combined example (from assignment spec)
python graph_analysis.py community_graph.gml --components 3 --plot C --simulate_failures 5 --output output.gml

# Full analysis + save
python graph_analysis.py sample_graph.gml --verify_homophily --verify_balanced_graph --output output.gml
```

---

## Approach

### Clustering Coefficients
Computed via `nx.clustering(graph)`. For a node v with neighbors N(v), the local clustering coefficient is `|{(u,w) : u,w ∈ N(v), (u,w) ∈ E}| / (k*(k-1)/2)` where k = |N(v)|.

### Neighborhood Overlap
For each edge (u, v): `NO = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|` where N(x) excludes the edge endpoints. A high overlap means u and v share many mutual friends.

### Girvan-Newman Partitioning
Iteratively removes the edge with the highest betweenness centrality (fraction of all shortest paths that pass through it). Stops when the desired number of components is reached.

### Homophily Test
For each connected node, computes the fraction of same-attribute neighbors minus the global base rate of that attribute. A one-sample t-test (alternative='greater') checks whether this excess is statistically significant (α = 0.05).

### Signed Graph Balance (BFS 2-coloring)
A signed graph is balanced iff its nodes can be 2-partitioned such that positive edges are within groups and negative edges are between groups. This is equivalent to BFS 2-coloring where: same color propagates across `sign=+1` edges, and opposite color propagates across `sign=-1` edges. A contradiction means imbalance.

### Edge Failure Simulation (`--simulate_failures k`)
Removes k randomly chosen edges from a copy of the graph and reports changes in: number of connected components, average shortest path length (largest component), and top-5 betweenness centrality nodes.

### Robustness Check (`--robustness_check k`)
Runs 10 independent rounds of k-edge removal. Reports aggregate statistics: mean/std component count, max/min component sizes, and the percentage of rounds in which the original cluster structure is preserved.

### Temporal Simulation
Reads a CSV with columns `source, target, timestamp, action` (add/remove). Groups events by timestamp into animation frames and displays the evolving graph using `matplotlib.animation.FuncAnimation`.

---

## Sample Input Files

| File | Purpose |
|------|---------|
| `sample_graph.gml` | 6-node balanced signed graph with `color` and `sign` attributes |
| `community_graph.gml` | 15-node graph with 3 dense cliques connected by bridge edges |
| `temporal_edges.csv` | 13-timestamp edge addition/removal sequence for animation |

---

## File Structure

```
graph_analysis.py       # Main program (all logic in one file)
sample_graph.gml        # Signed + colored test graph
community_graph.gml     # Three-community test graph
temporal_edges.csv      # Temporal edge log for animation
requirements.txt        # Python dependencies
README.md               # This file
```
