# ATLAS Game Network Analysis

This project analyzes the game of ATLAS using complex network theory and applies link prediction techniques with graph neural networks.

## Project Structure

- `data/`
  - `countries.csv` — Processed country data for network construction
  - `cities.csv` — Processed city data for network construction
  - `scrap_data.ipynb` — Notebook for scraping and processing raw data
- `code/`
  - `networks/`
    - `create_networks.ipynb` — Jupyter notebook to build networks from the processed CSVs
    - `dot/` — exported Graphviz DOT files (cities, countries, combined networks)
    - `graphml/` — exported GraphML files for interoperability with graph tools
  - `visualizations/`
    - `create_plots.ipynb` — Notebook to generate visualizations and analysis plots

## Goals

- Model the ATLAS game as a directed network using country and city names.
- Analyze network properties and structure.
- Apply graph neural networks for link prediction to suggest possible next moves in the game.

## Requirements

- Python 3.11+
- pandas
- networkx
- matplotlib
- pygraphviz (optional, used for hierarchical/dot layouts)
- jupyterlab / notebook

## Usage

1. Run `data/scrap_data.ipynb` (or open it in Jupyter) to regenerate or update `countries.csv` and `cities.csv`.
2. Open and run `code/networks/create_networks.ipynb` to build the networks. The notebook exports DOT files to `code/networks/dot/` and GraphML files to `code/networks/graphml/`.
3. Open and run `code/visualizations/create_plots.ipynb` to produce plots and visual summaries of the networks.

## About ATLAS

ATLAS is a word game where players take turns naming geographical locations (countries or cities), with each new name starting with the last letter of the previous one.

---

_This project is for research and educational purposes._
