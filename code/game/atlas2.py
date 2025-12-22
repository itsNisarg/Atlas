import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from seaborn import heatmap


class ATLASGame:
    """
    ATLAS game engine with graph-based strategies and interactive play.
    """

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------

    def __init__(self, places):
        self.places = sorted({p.strip().title() for p in places})
        self.graph = self._build_graph()

        self.used_places = set()
        self.current_place = None
        self.current_player = 1
        self.game_history = []

        self.parity_adv = self._safe_load(
            "code/analysis/parity/country_parity_adv.csv", "CountryParity"
        )
        self.hits_adv = self._safe_load("code/analysis/hits/country_hits.csv", "Adv")

    @staticmethod
    def _safe_load(path, col="Adv"):
        path = Path(path)
        if path.exists():
            return pd.read_csv(path, index_col=0, header=0).loc[:, col]
        return None

    # -------------------------------------------------
    # Graph construction
    # -------------------------------------------------

    def _build_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.places)

        start_map = defaultdict(list)
        for p in self.places:
            start_map[p[0].upper()].append(p)

        for a in self.places:
            for b in start_map[a[-1].upper()]:
                if a != b:
                    G.add_edge(a, b)

        return G

    # -------------------------------------------------
    # Game state helpers
    # -------------------------------------------------

    def reset(self):
        self.used_places.clear()
        self.current_place = None
        self.current_player = 1
        self.game_history.clear()

    def get_valid_moves(self, from_place=None):
        if from_place is None:
            from_place = self.current_place

        if from_place is None:
            return self.places

        return [
            p for p in self.graph.successors(from_place) if p not in self.used_places
        ]

    # -------------------------------------------------
    # Move execution
    # -------------------------------------------------

    def make_move(self, place):
        place = place.strip().title()

        if place not in self.places:
            return False, "Invalid place."
        if place in self.used_places:
            return False, "Place already used."
        if self.current_place:
            if place[0].upper() != self.current_place[-1].upper():
                return False, "Invalid starting letter."

        self.used_places.add(place)
        self.game_history.append((self.current_player, place))
        self.current_place = place

        if not self.get_valid_moves(place):
            return True, f"Player {self.current_player} wins!"

        self.current_player = 3 - self.current_player
        return True, "Move accepted."

    # -------------------------------------------------
    # AI strategies
    # -------------------------------------------------

    def get_ai_move(self, strategy="random"):
        moves = self.get_valid_moves()
        if not moves:
            return None

        if strategy == "random":
            return random.choice(moves)

        if strategy == "greedy_outdegree":
            return min(moves, key=lambda m: len(self.get_valid_moves(m)))

        # if strategy == "greedy_rare":
        #     freq = defaultdict(int)
        #     for p in self.places:
        #         if p not in self.used_places:
        #             freq[p[0]] += 1
        #     return min(moves, key=lambda m: freq[m[-1]])

        if strategy == "defensive":

            def score(m):
                future = self.get_valid_moves(m)
                depth2 = sum(len(list(self.graph.successors(f))) for f in future)
                return -len(future) + depth2  # depth2 can be avg node degree

            return max(moves, key=score)

        if strategy == "parity" and self.parity_adv is not None:
            return max(moves, key=lambda m: self.parity_adv.get(m, 0))

        if strategy == "hits" and self.hits_adv is not None:
            return max(moves, key=lambda m: self.hits_adv.get(m, 0))

        return random.choice(moves)

    # -------------------------------------------------
    # Interactive Human vs AI
    # -------------------------------------------------

    def play_interactive(self, ai_strategy="greedy_outdegree"):
        """
        Interactive Human vs AI game.
        Allows human to choose Player 1 or Player 2.
        """

        self.reset()

        print("\n" + "=" * 60)
        print("WELCOME TO ATLAS")
        print("=" * 60)
        print("Rules:")
        print("- Say a place starting with the last letter of the previous place")
        print("- No repeats")
        print("- Type 'quit' to exit")
        print("=" * 60)

        choice = input("Do you want to be Player 1 or Player 2? (1/2): ").strip()
        human_player = 1 if choice != "2" else 2
        ai_player = 3 - human_player

        self.current_player = 1

        while True:
            print("\n" + "-" * 50)
            print(f"Player {self.current_player}'s turn")

            if self.current_place:
                print(f"Current place: {self.current_place}")
                print(f"Required starting letter: {self.current_place[-1].upper()}")
            else:
                print("First move: any place allowed")

            # ------------------
            # Human turn
            # ------------------
            if self.current_player == human_player:
                move = input("Your move: ").strip()

                if move.lower() == "quit":
                    print("Game aborted.")
                    return

                success, message = self.make_move(move)
                print(message)

                if "wins!" in message:
                    return
                if not success:
                    continue

            # ------------------
            # AI turn
            # ------------------
            else:
                print("AI is thinking...")
                move = self.get_ai_move(ai_strategy)

                if move is None:
                    print("AI has no valid moves. You win!")
                    return

                print(f"AI plays: {move}")
                success, message = self.make_move(move)
                print(message)

                if "wins!" in message:
                    return

    # ======================
    # SIMULATION UTILITIES
    # ======================

    def simulate_game(self, s1, s2, starting_place=None):
        self.reset()

        if starting_place:
            ok, msg = self.make_move(starting_place)
            if not ok:
                return None

        while True:
            strat = s1 if self.current_player == 1 else s2
            move = self.get_ai_move(strat)
            if move is None:
                return 3 - self.current_player
            ok, msg = self.make_move(move)
            if "wins" in msg:
                return self.current_player

    def run_tournament(self, strategies, games=100):
        results = defaultdict(lambda: defaultdict(int))

        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1 :]:
                for _ in range(games):
                    winner = self.simulate_game(s1, s2)
                    results[s1]["games"] += 1
                    results[s2]["games"] += 1
                    if winner == 1:
                        results[s1]["wins"] += 1
                    else:
                        results[s2]["wins"] += 1

        return results


def main():
    import sys

    # ----------------------
    # Load country list
    # ----------------------
    try:
        countries = (
            pd.read_csv("data/countries.csv", header=0)["Country"].dropna().tolist()
        )
    except Exception as e:
        print("Failed to load country list:", e)
        sys.exit(1)

    game = ATLASGame(countries)

    strategies = [
        "random",
        "greedy_outdegree",
        "defensive",
        "parity",
        "hits",
    ]

    while True:
        print("\n" + "=" * 60)
        print("ATLAS GAME MENU")
        print("=" * 60)
        print("1. Interactive game (Human vs AI)")
        print("2. AI vs AI (single simulation)")
        print("3. Tournament (multiple strategies)")
        print("4. Strategy matchup analysis")
        print("5. Matrix Analysis")
        print("6. Exit")

        choice = input("\nEnter choice (1–6): ").strip()

        # -------------------------------------------------
        # 1. Interactive play
        # -------------------------------------------------
        if choice == "1":
            game.reset()
            game.play_interactive()

        # -------------------------------------------------
        # 2. Single AI vs AI game
        # -------------------------------------------------
        elif choice == "2":
            print("\nAvailable strategies:")
            print(", ".join(strategies))

            s1 = input("Player 1 strategy: ").strip() or "greedy_outdegree"
            s2 = input("Player 2 strategy: ").strip() or "defensive"

            if s1 not in strategies or s2 not in strategies:
                print("Invalid strategy name.")
                continue

            winner = game.simulate_game(s1, s2)
            for player, move in game.game_history:
                print(f"Player {player}: {move}")
            print(f"\nWinner: Player {winner}")

        # -------------------------------------------------
        # 3. Tournament
        # -------------------------------------------------
        elif choice == "3":
            print("\nAvailable strategies:")
            print(", ".join(strategies))

            selected = input("Enter comma-separated strategies (blank = all): ").strip()

            if selected:
                selected_strategies = [
                    s.strip() for s in selected.split(",") if s.strip() in strategies
                ]
            else:
                selected_strategies = strategies

            try:
                games = int(input("Games per matchup (default 100): ").strip() or "100")
            except ValueError:
                print("Invalid number.")
                continue

            results = game.run_tournament(selected_strategies, games)

            print("\nTournament Results:")
            print("-" * 40)
            for strat, stats in results.items():
                wins = stats.get("wins", 0)
                total = stats.get("games", 0)
                rate = wins / total if total else 0
                print(f"{strat:<20} {wins:>5}/{total:<5} ({rate*100:>5.1f}%)")

        # -------------------------------------------------
        # 4. Head-to-head analysis
        # -------------------------------------------------
        elif choice == "4":
            print("\nAvailable strategies:")
            print(", ".join(strategies))

            s1 = input("Player 1 strategy: ").strip()
            s2 = input("Player 2 strategy: ").strip()

            if s1 not in strategies or s2 not in strategies:
                print("Invalid strategy.")
                continue

            try:
                n = int(input("Number of games (default 50): ").strip() or "50")
            except ValueError:
                print("Invalid number.")
                continue

            p1_wins = 0
            for _ in range(n):
                winner = game.simulate_game(s1, s2)
                if winner == 1:
                    p1_wins += 1

            print(f"\n{s1} vs {s2}")
            print(f"Player 1 wins: {p1_wins}/{n} ({p1_wins/n*100:.1f}%)")
            print(f"Player 2 wins: {n-p1_wins}/{n} ({(n-p1_wins)/n*100:.1f}%)")

        # -------------------------------------------------
        # 5. Matrix Win Comparison
        # -------------------------------------------------
        elif choice == "5":
            print("\nAvailable strategies:")
            print(", ".join(strategies))

            matrix = np.zeros((len(strategies), len(strategies)), dtype=np.float16)

            for i, strat1 in enumerate(strategies):
                for j, strat2 in enumerate(strategies):
                    p1_wins = 0
                    wins = []
                    with ThreadPoolExecutor() as executor:
                        wins = executor.map(
                            game.simulate_game, [strat1] * 100, [strat2] * 100
                        )
                    wins = [1 if w == 1 else 0 for w in wins]
                    p1_wins = sum(wins)
                    print(f"{strat1} vs {strat2}")
                    print(f"Player 1 wins: {p1_wins}/100 ({p1_wins/100*100:.1f}%)")
                    print(
                        f"Player 2 wins: {100-p1_wins}/100 ({(100-p1_wins)/100*100:.1f}%)"
                    )
                    matrix[i, j] = p1_wins / 100

            print("\nWin matrix:")
            print(matrix)
            fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
            heatmap(
                matrix,
                square=True,
                cmap="RdYlGn",
                cbar_kws={"shrink": 0.8},
                vmin=0,
                vmax=1,
                annot=True,
                xticklabels=strategies,
                yticklabels=strategies,
                fmt=".2f",
                ax=ax,
            )
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            ax.set_title(
                "Win Percentage of Strategy i vs Strategy j \nFirst move decided by Strategy i",
                fontsize=16,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig("code/game/win_matrix.png")
        # -------------------------------------------------
        # 6. Exit
        # -------------------------------------------------
        elif choice == "6":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
