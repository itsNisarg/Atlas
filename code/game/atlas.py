import networkx as nx
from collections import defaultdict
import random
import pandas as pd


class ATLASGame:
    """
    Implementation of the ATLAS game with graph-theoretic analysis.
    """

    def __init__(self, places=None):
        """
        Initialize the game with a list of places.
        If no places provided, uses a default set of countries.
        """
        if places is None:
            # Default set of countries
            self.places = [
                "Afghanistan",
                "Albania",
                "Algeria",
                "Andorra",
                "Angola",
                "Argentina",
                "Armenia",
                "Australia",
                "Austria",
                "Azerbaijan",
                "Bahamas",
                "Bahrain",
                "Bangladesh",
                "Barbados",
                "Belarus",
                "Belgium",
                "Belize",
                "Benin",
                "Bhutan",
                "Bolivia",
                "Bosnia",
                "Botswana",
                "Brazil",
                "Brunei",
                "Bulgaria",
                "Burkina Faso",
                "Burundi",
                "Cambodia",
                "Cameroon",
                "Canada",
                "Chad",
                "Chile",
                "China",
                "Colombia",
                "Comoros",
                "Congo",
                "Costa Rica",
                "Croatia",
                "Cuba",
                "Cyprus",
                "Czechia",
                "Denmark",
                "Djibouti",
                "Dominica",
                "Ecuador",
                "Egypt",
                "El Salvador",
                "Eritrea",
                "Estonia",
                "Eswatini",
                "Ethiopia",
                "Fiji",
                "Finland",
                "France",
                "Gabon",
                "Gambia",
                "Georgia",
                "Germany",
                "Ghana",
                "Greece",
                "Grenada",
                "Guatemala",
                "Guinea",
                "Guyana",
                "Haiti",
                "Honduras",
                "Hungary",
                "Iceland",
                "India",
                "Indonesia",
                "Iran",
                "Iraq",
                "Ireland",
                "Israel",
                "Italy",
                "Jamaica",
                "Japan",
                "Jordan",
                "Kazakhstan",
                "Kenya",
                "Kiribati",
                "Kosovo",
                "Kuwait",
                "Kyrgyzstan",
                "Laos",
                "Latvia",
                "Lebanon",
                "Lesotho",
                "Liberia",
                "Libya",
                "Liechtenstein",
                "Lithuania",
                "Luxembourg",
                "Madagascar",
                "Malawi",
                "Malaysia",
                "Maldives",
                "Mali",
                "Malta",
                "Mauritania",
                "Mauritius",
                "Mexico",
                "Micronesia",
                "Moldova",
                "Monaco",
                "Mongolia",
                "Montenegro",
                "Morocco",
                "Mozambique",
                "Myanmar",
                "Namibia",
                "Nauru",
                "Nepal",
                "Netherlands",
                "New Zealand",
                "Nicaragua",
                "Niger",
                "Nigeria",
                "North Korea",
                "North Macedonia",
                "Norway",
                "Oman",
                "Pakistan",
                "Palau",
                "Palestine",
                "Panama",
                "Papua New Guinea",
                "Paraguay",
                "Peru",
                "Philippines",
                "Poland",
                "Portugal",
                "Qatar",
                "Romania",
                "Russia",
                "Rwanda",
                "Samoa",
                "San Marino",
                "Saudi Arabia",
                "Senegal",
                "Serbia",
                "Seychelles",
                "Sierra Leone",
                "Singapore",
                "Slovakia",
                "Slovenia",
                "Somalia",
                "South Africa",
                "South Korea",
                "South Sudan",
                "Spain",
                "Sri Lanka",
                "Sudan",
                "Suriname",
                "Sweden",
                "Switzerland",
                "Syria",
                "Taiwan",
                "Tajikistan",
                "Tanzania",
                "Thailand",
                "Togo",
                "Tonga",
                "Trinidad",
                "Tunisia",
                "Turkey",
                "Turkmenistan",
                "Tuvalu",
                "Uganda",
                "Ukraine",
                "United Arab Emirates",
                "United Kingdom",
                "United States",
                "Uruguay",
                "Uzbekistan",
                "Vanuatu",
                "Vatican",
                "Venezuela",
                "Vietnam",
                "Yemen",
                "Zambia",
                "Zimbabwe",
            ]
        else:
            self.places = places

        # Normalize places (strip whitespace, title case)
        # self.places = [p.strip().title() for p in self.places]

        # Build the game graph
        self.graph = self._build_graph()

        # Game state
        self.used_places = set()
        self.current_place = None
        self.current_player = 1
        self.game_history = []

        self.parity_adv = pd.read_csv(
            "code/analysis/parity/country_parity_adv.csv", index_col=0, header=0
        )
        self.hits_adv = pd.read_csv(
            "code/analysis/hits/country_hits.csv", index_col=0, header=0
        )

    def _build_graph(self):
        """Build directed graph where edges represent valid transitions."""
        G = nx.DiGraph()

        # Add all places as nodes
        for place in self.places:
            G.add_node(place)

        # Add edges: place A -> place B if B starts with last letter of A
        for place_a in self.places:
            last_letter = place_a[-1].upper()
            for place_b in self.places:
                if place_a != place_b and place_b[0].upper() == last_letter:
                    G.add_edge(place_a, place_b)

        return G

    def get_valid_moves(self, from_place=None):
        """Get all valid moves from the current position."""
        if from_place is None:
            if self.current_place is None:
                # Game hasn't started, all places are valid
                return [p for p in self.places if p not in self.used_places]
            from_place = self.current_place

        # Get successors that haven't been used
        valid = [
            p for p in self.graph.successors(from_place) if p not in self.used_places
        ]
        return valid

    def make_move(self, place):
        """
        Make a move in the game.
        Returns True if move is valid, False otherwise.
        """
        # Normalize input
        place = place.strip().title()

        # Check if place exists
        if place not in self.places:
            return False, f"'{place}' is not in the list of valid places."

        # Check if already used
        if place in self.used_places:
            return False, f"'{place}' has already been used."

        # Check if it's a valid move
        if self.current_place is None:
            # First move - any place is valid
            pass
        else:
            valid_moves = self.get_valid_moves()
            if place not in valid_moves:
                last_letter = self.current_place[-1].upper()
                return False, f"'{place}' doesn't start with '{last_letter}'."

        # Make the move
        self.used_places.add(place)
        self.game_history.append((self.current_player, place))
        self.current_place = place

        # Determine next player before checking game over
        next_player = 3 - self.current_player

        # Check if next player has valid moves
        valid_moves_for_next = [
            p for p in self.graph.successors(place) if p not in self.used_places
        ]

        if not valid_moves_for_next:
            # Next player has no moves, current player wins
            return (
                True,
                f"Player {next_player} has no valid moves. Player {self.current_player} wins!",
            )

        # Switch players
        self.current_player = next_player

        return (
            True,
            f"Valid move! Current place: {place}. Player {self.current_player}'s turn.",
        )

    def reset(self):
        """Reset the game."""
        self.used_places = set()
        self.current_place = None
        self.current_player = 1
        self.game_history = []

    def get_ai_move(self, strategy="random"):
        """
        Get an AI move based on the specified strategy.

        Strategies:
        - 'random': Choose randomly from valid moves
        - 'greedy_outdegree': Choose move with highest out-degree
        - 'greedy_rare': Choose move ending in rarest letter
        - 'defensive': Choose move with most available continuations
        """
        valid_moves = self.get_valid_moves()

        if not valid_moves:
            return None

        if strategy == "random":
            return random.choice(valid_moves)

        elif strategy == "greedy_outdegree":
            # Choose the move with highest out-degree (most options after)
            scores = []
            for move in valid_moves:
                future_moves = [
                    p for p in self.graph.successors(move) if p not in self.used_places
                ]
                scores.append((len(future_moves), move))
            scores.sort(reverse=True)
            return scores[0][1]

        elif strategy == "greedy_rare":
            # Choose move ending in rarest letter (forcing opponent)
            letter_freq = defaultdict(int)
            for place in self.places:
                if place not in self.used_places:
                    letter_freq[place[0].upper()] += 1

            scores = []
            for move in valid_moves:
                last_letter = move[-1].upper()
                rarity_score = letter_freq.get(last_letter, 0)
                scores.append((rarity_score, move))
            scores.sort()  # Lower is better (rarer)
            return scores[0][1]

        elif strategy == "defensive":
            # Similar to greedy_outdegree but considers depth
            scores = []
            for move in valid_moves:
                temp_used = self.used_places.copy()
                temp_used.add(move)

                # Count moves available after this move
                future_moves = [
                    p for p in self.graph.successors(move) if p not in temp_used
                ]

                # Also count second-level moves
                second_level = 0
                for fm in future_moves:
                    second_level += len(
                        [p for p in self.graph.successors(fm) if p not in temp_used]
                    )

                score = len(future_moves) + 0.5 * second_level
                scores.append((score, move))

            scores.sort(reverse=True)
            return scores[0][1]

        return random.choice(valid_moves)

    def play_interactive(self):
        """Play an interactive game against the computer."""
        print("=" * 60)
        print("Welcome to ATLAS!")
        print("=" * 60)
        print(f"Playing with {len(self.places)} places.")
        print(
            "Rules: Name a place starting with the last letter of the previous place."
        )
        print("No repeats allowed. Type 'quit' to exit, 'hint' for suggestions.\n")

        # Choose who goes first
        human_first = input("Do you want to go first? (y/n): ").lower().startswith("y")

        if not human_first:
            self.current_player = 2

        while True:
            print(f"\n{'='*60}")
            print(f"Player {self.current_player}'s turn")

            if self.current_place:
                print(f"Current place: {self.current_place}")
                print(f"Next place must start with: {self.current_place[-1].upper()}")
            else:
                print("First move - choose any place!")

            print(f"Places used: {len(self.used_places)}")

            # Human's turn
            if (human_first and self.current_player == 1) or (
                not human_first and self.current_player == 2
            ):

                valid_moves = self.get_valid_moves()
                print(f"You have {len(valid_moves)} valid moves available.")

                move = input("\nYour move: ").strip()

                if move.lower() == "quit":
                    print("Thanks for playing!")
                    break

                if move.lower() == "hint":
                    print(f"\nSome suggestions: {valid_moves[:5]}")
                    continue

                success, message = self.make_move(move)
                print(message)

                if "wins!" in message:
                    break

                if not success:
                    continue

            # AI's turn
            else:
                print("AI is thinking...")
                ai_move = self.get_ai_move(strategy="greedy_outdegree")

                if ai_move is None:
                    print(f"AI has no valid moves. You win!")
                    break

                success, message = self.make_move(ai_move)
                print(f"AI plays: {ai_move}")
                print(message)

                if "wins!" in message:
                    break

        # Show game history
        print("\n" + "=" * 60)
        print("Game History:")
        for i, (player, place) in enumerate(self.game_history, 1):
            print(f"{i}. Player {player}: {place}")


def analyze_graph_properties(game):
    """Analyze and print graph properties of the ATLAS game."""
    G = game.graph

    print("=" * 60)
    print("ATLAS Graph Analysis")
    print("=" * 60)

    print(f"\nBasic Properties:")
    print(f"  Number of places: {G.number_of_nodes()}")
    print(f"  Number of valid transitions: {G.number_of_edges()}")
    print(
        f"  Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}"
    )
    print(f"  Graph density: {nx.density(G):.4f}")

    # Out-degree distribution
    out_degrees = dict(G.out_degree())
    print(f"\nOut-Degree Statistics:")
    print(f"  Max out-degree: {max(out_degrees.values())}")
    print(f"  Min out-degree: {min(out_degrees.values())}")

    # Terminal nodes (dead ends)
    terminals = [node for node, deg in out_degrees.items() if deg == 0]
    print(f"\nTerminal nodes (dead ends): {len(terminals)}")
    if terminals:
        print(f"  Examples: {terminals[:5]}")

    # High out-degree nodes (safe moves)
    sorted_degrees = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMost flexible places (highest out-degree):")
    for place, degree in sorted_degrees[:5]:
        print(f"  {place}: {degree} options")

    # Strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    print(f"\nStrongly Connected Components: {len(sccs)}")
    scc_sizes = sorted([len(scc) for scc in sccs], reverse=True)
    print(f"  Largest SCC size: {scc_sizes[0]}")
    print(f"  SCC size distribution: {scc_sizes[:10]}")

    # Letter frequency analysis
    letter_start_freq = defaultdict(int)
    letter_end_freq = defaultdict(int)

    for place in game.places:
        letter_start_freq[place[0].upper()] += 1
        letter_end_freq[place[-1].upper()] += 1

    print(f"\nLetter Analysis:")
    print(
        f"  Most common starting letter: {max(letter_start_freq.items(), key=lambda x: x[1])}"
    )
    print(
        f"  Most common ending letter: {max(letter_end_freq.items(), key=lambda x: x[1])}"
    )

    # Dangerous letters (high in-degree, low out-degree)
    print(f"\nMost dangerous ending letters (few places start with them):")
    for letter in sorted(letter_end_freq.keys()):
        if letter_end_freq[letter] > 0:
            ratio = letter_start_freq.get(letter, 0) / letter_end_freq[letter]
            if ratio < 0.5:
                print(
                    f"  {letter}: {letter_end_freq[letter]} places end with it, "
                    f"{letter_start_freq.get(letter, 0)} start with it (ratio: {ratio:.2f})"
                )


def simulate_game(
    game, strategy_p1="random", strategy_p2="random", verbose=False, starting_place=None
):
    """
    Simulate a complete game between two AI players.

    Args:
        game: ATLASGame instance
        strategy_p1: Strategy for player 1
        strategy_p2: Strategy for player 2
        verbose: If True, print move-by-move details
        starting_place: Optional starting place (if None, AI chooses)

    Returns:
        Dictionary with game results
    """
    game.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulating: Player 1 ({strategy_p1}) vs Player 2 ({strategy_p2})")
        print(f"{'='*60}\n")

    move_count = 0
    winner = None

    # First move
    if starting_place:
        success, message = game.make_move(starting_place)
        if not success:
            return {"error": message}
        move_count += 1
        if verbose:
            print(f"Move {move_count}: Player 1 starts with {starting_place}")

        # Check if game ended on first move
        if "wins!" in message:
            # Extract winner from message
            if "Player 1 wins!" in message:
                winner = 1
            else:
                winner = 2
            if verbose:
                print(f"\n{message}\n")

    # Play until game ends
    while winner is None:
        current_strategy = strategy_p1 if game.current_player == 1 else strategy_p2
        player_making_move = game.current_player  # Capture BEFORE making move

        ai_move = game.get_ai_move(strategy=current_strategy)

        if ai_move is None:
            # Current player has no valid moves available from get_ai_move
            winner = 3 - game.current_player
            if verbose:
                print(f"\nPlayer {game.current_player} has no valid moves!")
                print(f"Player {winner} wins!\n")
            break

        success, message = game.make_move(ai_move)
        move_count += 1

        if verbose:
            print(f"Move {move_count}: Player {player_making_move} plays {ai_move}")

        if "wins!" in message:
            # Extract winner from message
            if "Player 1 wins!" in message:
                winner = 1
            else:
                winner = 2
            if verbose:
                print(f"\n{message}\n")
            break

    return {
        "winner": winner,
        "moves": move_count,
        "history": game.game_history.copy(),
        "final_place": game.current_place,
        "places_used": len(game.used_places),
    }


def run_tournament(game, strategies, num_games=100, verbose=False):
    """
    Run a tournament between different strategies.

    Args:
        game: ATLASGame instance
        strategies: List of strategy names to compete
        num_games: Number of games per matchup
        verbose: Print detailed results

    Returns:
        Dictionary with tournament results
    """
    results = defaultdict(lambda: defaultdict(int))
    matchup_details = []

    print(f"\n{'='*60}")
    print(f"Running Tournament: {num_games} games per matchup")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"{'='*60}\n")

    # All pairs of strategies
    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies):
            if i >= j:  # Avoid duplicate matchups and self-play for now
                continue

            print(f"Matchup: {strat1} vs {strat2}")

            wins_p1 = 0
            wins_p2 = 0
            total_moves = []

            for game_num in range(num_games):
                # Player 1 uses strat1, Player 2 uses strat2
                result = simulate_game(game, strat1, strat2, verbose=False)

                if result["winner"] == 1:
                    wins_p1 += 1
                else:
                    wins_p2 += 1

                total_moves.append(result["moves"])

            avg_moves = sum(total_moves) / len(total_moves)

            print(
                f"  {strat1} wins: {wins_p1}/{num_games} ({wins_p1/num_games*100:.1f}%)"
            )
            print(
                f"  {strat2} wins: {wins_p2}/{num_games} ({wins_p2/num_games*100:.1f}%)"
            )
            print(f"  Average game length: {avg_moves:.1f} moves\n")

            results[strat1]["wins"] += wins_p1
            results[strat1]["losses"] += wins_p2
            results[strat2]["wins"] += wins_p2
            results[strat2]["losses"] += wins_p1

            matchup_details.append(
                {
                    "player1": strat1,
                    "player2": strat2,
                    "p1_wins": wins_p1,
                    "p2_wins": wins_p2,
                    "games": num_games,
                    "avg_moves": avg_moves,
                }
            )

    # Print summary
    print(f"\n{'='*60}")
    print("Tournament Summary")
    print(f"{'='*60}\n")

    strategy_stats = []
    for strategy in strategies:
        total_games = results[strategy]["wins"] + results[strategy]["losses"]
        if total_games > 0:
            win_rate = results[strategy]["wins"] / total_games
            strategy_stats.append(
                (
                    strategy,
                    results[strategy]["wins"],
                    results[strategy]["losses"],
                    win_rate,
                )
            )

    strategy_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"{'Strategy':<20} {'Wins':<8} {'Losses':<8} {'Win Rate':<10}")
    print("-" * 60)
    for strategy, wins, losses, win_rate in strategy_stats:
        print(f"{strategy:<20} {wins:<8} {losses:<8} {win_rate*100:>6.1f}%")

    return {
        "overall_results": dict(results),
        "matchup_details": matchup_details,
        "rankings": strategy_stats,
    }


def analyze_strategy_matchup(game, strategy_p1, strategy_p2, num_games=50):
    """
    Detailed analysis of a specific strategy matchup.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {strategy_p1} vs {strategy_p2}")
    print(f"{'='*60}\n")

    p1_wins = 0
    p2_wins = 0
    game_lengths = []
    p1_first_moves = defaultdict(int)
    winning_first_moves = defaultdict(int)

    for _ in range(num_games):
        result = simulate_game(game, strategy_p1, strategy_p2)

        game_lengths.append(result["moves"])

        if result["winner"] == 1:
            p1_wins += 1
        else:
            p2_wins += 1

        # Track first moves
        if result["history"]:
            first_move = result["history"][0][1]
            p1_first_moves[first_move] += 1
            if result["winner"] == 1:
                winning_first_moves[first_move] += 1

    print(f"Results over {num_games} games:")
    print(f"  {strategy_p1} (P1) wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"  {strategy_p2} (P2) wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")

    print(f"\nGame Length Statistics:")
    print(f"  Average: {sum(game_lengths)/len(game_lengths):.1f} moves")
    print(f"  Shortest: {min(game_lengths)} moves")
    print(f"  Longest: {max(game_lengths)} moves")

    print(f"\nMost Common First Moves by {strategy_p1}:")
    sorted_first = sorted(p1_first_moves.items(), key=lambda x: x[1], reverse=True)
    for place, count in sorted_first[:5]:
        win_rate = winning_first_moves[place] / count if count > 0 else 0
        print(f"  {place}: {count} times (won {win_rate*100:.1f}%)")

    # Player 1 advantage analysis
    p1_advantage = (p1_wins - p2_wins) / num_games
    print(f"\nPlayer 1 Advantage: {p1_advantage*100:+.1f}%")
    if abs(p1_advantage) < 0.1:
        print("  → Matchup appears balanced")
    elif p1_advantage > 0:
        print("  → Player 1 (first mover) has advantage")
    else:
        print("  → Player 2 (second mover) has advantage")


def compare_opening_moves(game, strategy="greedy_outdegree", num_simulations=20):
    """
    Compare different opening moves to find the best starting places.
    """
    print(f"\n{'='*60}")
    print(f"Opening Move Analysis (Strategy: {strategy})")
    print(f"{'='*60}\n")

    opening_stats = []

    # Get some candidate opening moves (highest out-degree)
    out_degrees = dict(game.graph.out_degree())
    candidate_openings = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]

    print(f"Testing top {len(candidate_openings)} places by out-degree...\n")

    for place, out_degree in candidate_openings:
        wins = 0
        losses = 0
        game_lengths = []

        for _ in range(num_simulations):
            result = simulate_game(
                game, strategy, strategy, verbose=False, starting_place=place
            )

            game_lengths.append(result["moves"])
            if result["winner"] == 1:
                wins += 1
            else:
                losses += 1

        win_rate = wins / (wins + losses)
        avg_length = sum(game_lengths) / len(game_lengths)

        opening_stats.append(
            {
                "place": place,
                "out_degree": out_degree,
                "p1_wins": wins,
                "p2_wins": losses,
                "win_rate": win_rate,
                "avg_length": avg_length,
            }
        )

    # Sort by win rate
    opening_stats.sort(key=lambda x: x["win_rate"], reverse=True)

    print(
        f"{'Opening Move':<25} {'Out-Deg':<8} {'P1 Wins':<10} {'Win Rate':<10} {'Avg Length':<12}"
    )
    print("-" * 80)
    for stat in opening_stats[:15]:
        print(
            f"{stat['place']:<25} {stat['out_degree']:<8} "
            f"{stat['p1_wins']}/{num_simulations:<7} "
            f"{stat['win_rate']*100:>6.1f}% {stat['avg_length']:>10.1f}"
        )

    print(f"\nBest opening moves (by P1 win rate):")
    for i, stat in enumerate(opening_stats[:3], 1):
        print(f"  {i}. {stat['place']} (wins {stat['win_rate']*100:.1f}% as P1)")

    print(f"\nWorst opening moves (by P1 win rate):")
    for i, stat in enumerate(opening_stats[-3:], 1):
        print(f"  {i}. {stat['place']} (wins {stat['win_rate']*100:.1f}% as P1)")


if __name__ == "__main__":
    # Create game
    game = ATLASGame(
        places=pd.read_csv("data/countries.csv", header=0)["Country"].values.tolist()
    )

    # Analyze graph
    analyze_graph_properties(game)

    print("\n" * 2)
    print("Choose mode:")
    print("1. Interactive game (Human vs AI)")
    print("2. AI vs AI simulation (single game)")
    print("3. Tournament (multiple strategies)")
    print("4. Strategy matchup analysis")
    print("5. Opening move analysis")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        game.reset()
        game.play_interactive()

    elif choice == "2":
        print(
            "\nAvailable strategies: random, greedy_outdegree, greedy_rare, defensive, parity, hits"
        )
        strat1 = input("Player 1 strategy: ").strip() or "greedy_outdegree"
        strat2 = input("Player 2 strategy: ").strip() or "greedy_rare"

        result = simulate_game(game, strat1, strat2, verbose=True)

        print(f"\nFinal Statistics:")
        print(f"  Winner: Player {result['winner']}")
        print(f"  Total moves: {result['moves']}")
        print(f"  Places used: {result['places_used']}/{len(game.places)}")

    elif choice == "3":
        strategies = [
            "random",
            "greedy_outdegree",
            "greedy_rare",
            "defensive",
            "parity",
            "hits",
        ]
        num_games = int(input("Games per matchup (default 100): ").strip() or "100")

        results = run_tournament(game, strategies, num_games=num_games)

    elif choice == "4":
        print(
            "\nAvailable strategies: random, greedy_outdegree, greedy_rare, defensive, parity, hits"
        )
        strat1 = input("Player 1 strategy: ").strip() or "greedy_outdegree"
        strat2 = input("Player 2 strategy: ").strip() or "greedy_rare"
        num_games = int(input("Number of games (default 50): ").strip() or "50")

        analyze_strategy_matchup(game, strat1, strat2, num_games=num_games)

    elif choice == "5":
        print(
            "\nAvailable strategies: random, greedy_outdegree, greedy_rare, defensive, parity, hits"
        )
        strategy = (
            input("Strategy to test (default greedy_outdegree): ").strip()
            or "greedy_outdegree"
        )
        num_sims = int(input("Simulations per opening (default 20): ").strip() or "20")

        compare_opening_moves(game, strategy=strategy, num_simulations=num_sims)

    else:
        print("Invalid choice!")
