import networkx as nx
from collections import defaultdict
import random

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
                "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
                "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
                "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
                "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
                "Bosnia", "Botswana", "Brazil", "Brunei", "Bulgaria",
                "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
                "Chad", "Chile", "China", "Colombia", "Comoros",
                "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus",
                "Czechia", "Denmark", "Djibouti", "Dominica", "Ecuador",
                "Egypt", "El Salvador", "Eritrea", "Estonia", "Eswatini",
                "Ethiopia", "Fiji", "Finland", "France", "Gabon",
                "Gambia", "Georgia", "Germany", "Ghana", "Greece",
                "Grenada", "Guatemala", "Guinea", "Guyana", "Haiti",
                "Honduras", "Hungary", "Iceland", "India", "Indonesia",
                "Iran", "Iraq", "Ireland", "Israel", "Italy",
                "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
                "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos",
                "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
                "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
                "Malaysia", "Maldives", "Mali", "Malta", "Mauritania",
                "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco",
                "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
                "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
                "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia",
                "Norway", "Oman", "Pakistan", "Palau", "Palestine",
                "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
                "Poland", "Portugal", "Qatar", "Romania", "Russia",
                "Rwanda", "Samoa", "San Marino", "Saudi Arabia", "Senegal",
                "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia",
                "Slovenia", "Somalia", "South Africa", "South Korea", "South Sudan",
                "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
                "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania",
                "Thailand", "Togo", "Tonga", "Trinidad", "Tunisia",
                "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
                "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
                "Vanuatu", "Vatican", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
            ]
        else:
            self.places = places
        
        # Normalize places (strip whitespace, title case)
        self.places = [p.strip().title() for p in self.places]
        
        # Build the game graph
        self.graph = self._build_graph()
        
        # Game state
        self.used_places = set()
        self.current_place = None
        self.current_player = 1
        self.game_history = []
        
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
        valid = [p for p in self.graph.successors(from_place) 
                if p not in self.used_places]
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
        
        # Check if game is over (no valid moves left)
        if not self.get_valid_moves():
            winner = 3 - self.current_player  # Other player wins
            return True, f"Player {self.current_player} has no valid moves. Player {winner} wins!"
        
        # Switch players
        self.current_player = 3 - self.current_player
        
        return True, f"Valid move! Current place: {place}. Player {self.current_player}'s turn."
    
    def reset(self):
        """Reset the game."""
        self.used_places = set()
        self.current_place = None
        self.current_player = 1
        self.game_history = []
    
    def get_ai_move(self, strategy='random'):
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
        
        if strategy == 'random':
            return random.choice(valid_moves)
        
        elif strategy == 'greedy_outdegree':
            # Choose the move with highest out-degree (most options after)
            scores = []
            for move in valid_moves:
                future_moves = [p for p in self.graph.successors(move) 
                              if p not in self.used_places]
                scores.append((len(future_moves), move))
            scores.sort(reverse=True)
            return scores[0][1]
        
        elif strategy == 'greedy_rare':
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
        
        elif strategy == 'defensive':
            # Similar to greedy_outdegree but considers depth
            scores = []
            for move in valid_moves:
                temp_used = self.used_places.copy()
                temp_used.add(move)
                
                # Count moves available after this move
                future_moves = [p for p in self.graph.successors(move) 
                              if p not in temp_used]
                
                # Also count second-level moves
                second_level = 0
                for fm in future_moves:
                    second_level += len([p for p in self.graph.successors(fm) 
                                       if p not in temp_used])
                
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
        print("Rules: Name a place starting with the last letter of the previous place.")
        print("No repeats allowed. Type 'quit' to exit, 'hint' for suggestions.\n")
        
        # Choose who goes first
        human_first = input("Do you want to go first? (y/n): ").lower().startswith('y')
        
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
            if (human_first and self.current_player == 1) or \
               (not human_first and self.current_player == 2):
                
                valid_moves = self.get_valid_moves()
                print(f"You have {len(valid_moves)} valid moves available.")
                
                move = input("\nYour move: ").strip()
                
                if move.lower() == 'quit':
                    print("Thanks for playing!")
                    break
                
                if move.lower() == 'hint':
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
                ai_move = self.get_ai_move(strategy='greedy_outdegree')
                
                if ai_move is None:
                    print(f"AI has no valid moves. You win!")
                    break
                
                success, message = self.make_move(ai_move)
                print(f"AI plays: {ai_move}")
                print(message)
                
                if "wins!" in message:
                    break
        
        # Show game history
        print("\n" + "="*60)
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
    print(f"  Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")
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
    print(f"  Most common starting letter: {max(letter_start_freq.items(), key=lambda x: x[1])}")
    print(f"  Most common ending letter: {max(letter_end_freq.items(), key=lambda x: x[1])}")
    
    # Dangerous letters (high in-degree, low out-degree)
    print(f"\nMost dangerous ending letters (few places start with them):")
    for letter in sorted(letter_end_freq.keys()):
        if letter_end_freq[letter] > 0:
            ratio = letter_start_freq.get(letter, 0) / letter_end_freq[letter]
            if ratio < 0.5:
                print(f"  {letter}: {letter_end_freq[letter]} places end with it, "
                      f"{letter_start_freq.get(letter, 0)} start with it (ratio: {ratio:.2f})")


if __name__ == "__main__":
    # Create game
    game = ATLASGame()
    
    # Analyze graph
    analyze_graph_properties(game)
    
    # Play interactive game
    print("\n" * 2)
    play_interactive = input("Would you like to play an interactive game? (y/n): ")
    if play_interactive.lower().startswith('y'):
        game.reset()
        game.play_interactive()