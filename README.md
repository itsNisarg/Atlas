# ATLAS Game Network Analysis

This project analyzes the game of ATLAS using complex network theory and applies link prediction techniques with graph neural networks.

## Project Structure

- `data/`
  - `countries.csv` — Processed country data for network construction
  - `cities.csv` — Processed city data for network construction
  - `scrap_data.ipynb` — Notebook for scraping and processing raw data
- `code/networks/`
  - `network.ipynb` — Jupyter notebook for building and visualizing ATLAS networks

## Goals

- Model the ATLAS game as a directed network using country and city names.
- Analyze network properties and structure.
- Apply graph neural networks for link prediction to suggest possible next moves in the game.

## Requirements

- Python 3.11+
- pandas
- networkx
- matplotlib
- pygraphviz (for hierarchical layouts)
- Jupyter Notebook

## Usage

1. Run `data/scrap_data.ipynb` to generate the latest `countries.csv` and `cities.csv`.
2. Open and run `code/networks/network.ipynb` to build, visualize, and analyze the ATLAS networks.

## About ATLAS

ATLAS is a word game where players take turns naming geographical locations (countries or cities), with each new name starting with the last letter of the previous one.

---

_This project is for research and educational purposes._
