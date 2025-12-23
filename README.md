# ATLAS Game Network Analysis

This project analyzes the geographical word game **ATLAS** using complex network theory. It models countries and cities as nodes in a directed graph, where an edge exists if one location's name ends with the letter that the next location's name begins with.
<p align="center"><img src="code/visualizations/country.png" alt="Strategy Comparison" width="500"></p>

The project encompasses data scraping, network construction, deep structural analysis, and an interactive game engine with advanced AI strategies.

---

## Table of Contents

- [About ATLAS](#about-atlas)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Project Workflow](#project-workflow)

---

## About ATLAS

ATLAS is a classic word game where players name geographical locations. Each name must start with the last letter of the previous one (e.g., **A**tlanti**c** → **C**anad**a** → **A**lbania). The game ends when a player cannot provide a valid, previously unused location.

This project uses network science to analyze the game's structure and develop optimal playing strategies.

---

## Repository Structure

```
Atlas/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── cities.csv           # Processed city data
│   ├── countries.csv         # Processed country data
│   └── scrap_data.ipynb      # Data collection and cleaning notebook
│
└── code/
    ├── analysis/
    │   ├── analyse_nets.ipynb           # Main analysis notebook (metrics & centrality)
    │   ├── analyse_nodes.ipynb          # Node-level analysis
    │   ├── analyse_edges.ipynb          # Edge-level analysis
    │   ├── winning_paths.txt            # Winning path analysis results
    │   ├── pagerank/                    # PageRank scores (city, country, combined)
    │   ├── hits/                        # HITS hubs & authorities scores
    │   ├── degree/                      # Degree advantage metrics
    │   ├── between/                     # Betweenness centrality scores
    │   ├── scc/                         # Strongly connected components data
    │   ├── trophiclvl/                  # Trophic level metrics
    │   ├── parity/                      # Parity advantage metrics
    │   └── avg_neighbour/               # Average neighbor degree metrics
    │
    ├── networks/
    │   ├── create_networks.ipynb        # Network construction notebook
    │   ├── dot/                         # Exported graphs in DOT format
    │   └── graphml/                     # Exported graphs in GraphML format
    │
    ├── visualizations/
    │   ├── create_plots.ipynb           # Plotting and visualization notebook
    │   ├── city_net.png                 # City network visualization
    │   ├── country_net.png              # Country network visualization
    │   ├── combined_net.png             # Combined network visualization
    │   ├── combined_net.html            # Interactive network visualization
    │   └── lib/                         # Visualization libraries
    │
    ├── game/
    │   ├── atlas.py                     # Interactive game engine
    │   ├── win_matrix.png               # Strategy performance heatmap
    │   ├── City_win_matrix.png          # City-only game results
    │   ├── Country_win_matrix.png       # Country-only game results
    │   └── Combined_win_matrix.png      # Combined game results
    │
    ├── bonus/
    │   └── link_pred.ipynb              # Link prediction analysis
    │
    ├── community/
    │   └── detect_community.ipynb       # Community detection analysis
    │
    └── node2vec/
        └── embeddings.ipynb             # Node2vec embeddings
```

---

## Key Features

- **Multi-Level Modeling**: Analyze games played with just countries, just cities, or a combined global dataset
- **Comprehensive Graph Metrics**:
  - PageRank & HITS (Hubs and Authorities)
  - Betweenness Centrality
  - Strongly Connected Components (SCC)
  - Trophic Levels & Parity Analysis
  - Average Neighbor Degree
- **Advanced AI Strategies**: 
  - Greedy Out-degree
  - Defensive (minimize opponent's options)
  - Parity-based strategies
  - HITS Hubs-based approaches
  - PageRank-optimized play
- **Interactive Game Engine**: 
  - Human vs AI gameplay
  - AI vs AI simulations
  - Full tournament mode to compare strategies
- **Rich Visualizations**: Network graphs, degree distributions, and strategy performance heatmaps

<p align="center"><img src="code/game/win_matrix.png" alt="Strategy Performance Matrix" width="350"></p>

---

## Environment Setup

### Prerequisites

- **Python 3.11 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Atlas
   ```

2. **Create a virtual environment** (recommended)
   
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies

- **pygraphviz**: Required for high-quality DOT graph exports. Installation can be platform-specific:
  - **Windows**: May require [Graphviz](https://graphviz.org/download/) to be installed separately
  - **macOS**: `brew install graphviz` then `pip install pygraphviz`
  - **Linux**: `sudo apt-get install graphviz graphviz-dev` then `pip install pygraphviz`

### Key Dependencies

The project uses the following major libraries (see [requirements.txt](requirements.txt) for full list):

- `networkx` (3.6.1): Graph algorithms and data structures
- `pandas` (2.3.3), `numpy` (1.26.4): Data manipulation and numerical computing
- `matplotlib` (3.10.8), `seaborn` (0.13.2): Visualization and plotting
- `scipy` (1.16.3): Scientific computing
- `pyvis` (0.3.2): Interactive network visualizations
- `node2vec` (0.5.0), `gensim` (4.4.0): Graph embedding techniques
- `python-louvain` (0.16): Community detection
- `jupyterlab`: For running analysis notebooks

---

## Usage

### Quick Start

```bash
# Activate virtual environment (if not already activated)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run the game
python code/game/atlas.py
```

### Project Workflow

Follow these steps to reproduce the full analysis pipeline:

1. **Data Collection** (Optional - data already provided)
   ```bash
   jupyter lab data/scrap_data.ipynb
   ```
   Updates `cities.csv` and `countries.csv` from online sources

2. **Network Construction**
   ```bash
   jupyter lab code/networks/create_networks.ipynb
   ```
   Builds directed graphs for countries, cities, and combined datasets
   Exports graphs in DOT and GraphML formats

3. **Network Analysis** ⚠️ **Important Step**
   ```bash
   jupyter lab code/analysis/analyse_nets.ipynb
   ```
   Calculates centrality metrics, PageRank, HITS, SCC, trophic levels, etc.
   Generates CSV files in `code/analysis/*/` subdirectories
   **These CSV files are required by the game engine for AI strategies**

4. **Visualization**
   ```bash
   jupyter lab code/visualizations/create_plots.ipynb
   ```
   Generates network visualizations and statistical plots

5. **Play the Game**
   ```bash
   python code/game/atlas.py
   ```
   Choose from:
   - **Interactive Mode**: Play against AI
   - **Simulation Mode**: Watch AI vs AI games
   - **Tournament Mode**: Compare all strategies

### Game Modes

The game engine (`atlas.py`) offers three modes:

- **Human vs AI**: Test your skills against different AI strategies
- **AI vs AI**: Observe how different strategies perform against each other
- **Tournament**: Run a complete round-robin tournament to identify the best strategy

---

## Additional Analysis

Beyond the core workflow, the repository includes:

- **Community Detection** (`code/community/detect_community.ipynb`): Identify clusters in the network
- **Node Embeddings** (`code/node2vec/embeddings.ipynb`): Learn vector representations of locations
- **Link Prediction** (`code/bonus/link_pred.ipynb`): Predict potential new edges

---

_This project is for research and educational purposes._
