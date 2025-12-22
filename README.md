# ATLAS Game Network Analysis

This project analyzes the geographical word game **ATLAS** using complex network theory. It models countries and cities as nodes in a directed graph, where an edge exists if one location's name ends with the letter that the next location's name begins with.

The project encompasses data scraping, network construction, deep structural analysis, and an interactive game engine with advanced strategies.

## Project Structure

- **`data/`**: Datasets and collection scripts.
  - `countries.csv`, `cities.csv`: Processed geographical data.
  - `scrap_data.ipynb`: Scraping and cleaning raw data from various sources.
- **`code/`**: The core logic of the project.
  - **`networks/`**: Network construction module.
    - `create_networks.ipynb`: Builds directed graphs for countries, cities, and combined datasets.
    - `dot/` & `graphml/`: Exported graphs in standard formats for interoperability.
  - **`analysis/`**: Comprehensive network analysis tools.
    - `analyse_nets.ipynb`: Main analysis notebook calculating various centrality and structural metrics.
    - Subdirectories (e.g., `pagerank/`, `hits/`, `degree/`, `scc/`, `trophiclvl/`, `parity/`): Store CSV results for specific graph metrics used to drive strategies.
  - **`visualizations/`**: Visual outputs and plotting scripts.
    - `create_plots.ipynb`: Generates high-resolution network visualizations and data plots.
    - `city_net.png`, `country_net.png`, etc.: Pre-generated visualizations.
  - **`game/`**: The ATLAS game application.
    - `atlas.py`: Interactive game engine. Features include Human vs AI, AI vs AI simulations, and Strategy Tournaments.
    - `win_matrix.png`: Analysis of strategy performance.

## Key Features

- **Multi-Level Modeling**: Analyze games played with just countries, just cities, or a combined global dataset.
- **Graph Metrics**: Deep dive into PageRank, HITS (Hubs and Authorities), Betweenness Centrality, Strong Connectivity (SCC), and Trophic Levels.
- **Advanced AI Strategies**: Strategies based on network properties, including Greedy Out-degree, Defensive, Parity-based, and Hubs-based approaches.
- **Interactive Play**: Play against different AI personas or run massive tournaments to find the most dominant strategy.

## Requirements

- Python 3.11+
- `networkx`: Graph algorithms and structure.
- `pandas`, `numpy`: Data manipulation.
- `matplotlib`, `seaborn`: Visualization and plotting.
- `pygraphviz`: (Optional) For high-quality DOT layout exports.
- `jupyterlab`: To run the analysis and scraping notebooks.

## Usage

1. **Prepare Data**: Run `data/scrap_data.ipynb` to update the geographical source files.
2. **Build Networks**: Run `code/networks/create_networks.ipynb` to generate the graph files.
3. **Analyze & Export**: Run `code/analysis/analyse_nets.ipynb`. This step is crucial as it generates the CSV files in `code/analysis/*/` that the game engine uses for its advanced strategies.
4. **Play the Game**:
   ```bash
   python code/game/atlas.py
   ```
   Choose between interactive mode, single simulations, or full tournaments.

## About ATLAS

ATLAS is a classic word game where players name geographical locations. Each name must start with the last letter of the previous one (e.g., **A**tlanti**c** -> **C**anad**a** -> **A**lbania). The game ends when a player cannot provide a valid, previously unused location.

---

_This project is for research and educational purposes._
