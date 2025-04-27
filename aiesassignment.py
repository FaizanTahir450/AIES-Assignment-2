import time
import copy
from typing import List, Tuple, Optional
import random

class TicTacToe:
    """Tic-Tac-Toe game implementation with AI using Minimax and Alpha-Beta Pruning."""

    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.nodes_evaluated_minimax = 0
        self.nodes_evaluated_alpha_beta = 0

    def print_board(self) -> None:
        """Prints the current state of the game board."""
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)

    def is_winner(self, player: str) -> bool:
        """Checks if the specified player has won."""
        # Check rows
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        # Check columns
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        # Check diagonals
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        return False

    def is_board_full(self) -> bool:
        """Checks if the board is full (draw condition)."""
        return all(cell != ' ' for row in self.board for cell in row)

    def is_game_over(self) -> bool:
        """Checks if the game is over (win or draw)."""
        return self.is_winner('X') or self.is_winner('O') or self.is_board_full()

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Returns a list of empty cell coordinates."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def make_move(self, row: int, col: int, player: str) -> bool:
        """Attempts to make a move for the specified player."""
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == ' ':
            self.board[row][col] = player
            return True
        return False

    def minimax(self, depth: int, is_maximizing: bool) -> int:
        """Standard Minimax algorithm implementation."""
        self.nodes_evaluated_minimax += 1

        if self.is_winner('O'):
            return 10 - depth
        if self.is_winner('X'):
            return depth - 10
        if self.is_board_full():
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'O'
                score = self.minimax(depth + 1, False)
                self.board[row][col] = ' '
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'X'
                score = self.minimax(depth + 1, True)
                self.board[row][col] = ' '
                best_score = min(best_score, score)
            return best_score

    def alpha_beta(self, depth: int, is_maximizing: bool, alpha: float, beta: float) -> int:
        """Minimax with Alpha-Beta Pruning implementation."""
        self.nodes_evaluated_alpha_beta += 1

        if self.is_winner('O'):
            return 10 - depth
        if self.is_winner('X'):
            return depth - 10
        if self.is_board_full():
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'O'
                score = self.alpha_beta(depth + 1, False, alpha, beta)
                self.board[row][col] = ' '
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return best_score
        else:
            best_score = float('inf')
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'X'
                score = self.alpha_beta(depth + 1, True, alpha, beta)
                self.board[row][col] = ' '
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return best_score

    def get_best_move_minimax(self) -> Optional[Tuple[int, int]]:
        """Finds the best move using standard Minimax."""
        best_score = float('-inf')
        best_move = None
        self.nodes_evaluated_minimax = 0

        start_time = time.time()
        for row, col in self.get_empty_cells():
            self.board[row][col] = 'O'
            score = self.minimax(0, False)
            self.board[row][col] = ' '
            if score > best_score:
                best_score = score
                best_move = (row, col)
        end_time = time.time()

        return best_move, self.nodes_evaluated_minimax, end_time - start_time

    def get_best_move_alpha_beta(self) -> Optional[Tuple[int, int]]:
        """Finds the best move using Alpha-Beta Pruning."""
        best_score = float('-inf')
        best_move = None
        self.nodes_evaluated_alpha_beta = 0

        start_time = time.time()
        for row, col in self.get_empty_cells():
            self.board[row][col] = 'O'
            score = self.alpha_beta(0, False, float('-inf'), float('inf'))
            self.board[row][col] = ' '
            if score > best_score:
                best_score = score
                best_move = (row, col)
        end_time = time.time()

        return best_move, self.nodes_evaluated_alpha_beta, end_time - start_time

    def play_game(self):
        """Main game loop for human vs AI."""
        print("Welcome to Tic-Tac-Toe! You are X, AI is O.")
        print("Enter moves as 'row col' (0-2 for each).")

        while not self.is_game_over():
            self.print_board()

            # Human move
            if self.current_player == 'X':
                try:
                    row, col = map(int, input("Your move (row col): ").split())
                    if not self.make_move(row, col, 'X'):
                        print("Invalid move! Try again.")
                        continue
                except ValueError:
                    print("Invalid input! Use format 'row col'.")
                    continue

            # AI move
            else:
                print("AI thinking...")
                # Compare both algorithms
                move_minimax, nodes_minimax, time_minimax = self.get_best_move_minimax()
                move_alpha_beta, nodes_alpha_beta, time_alpha_beta = self.get_best_move_alpha_beta()

                print(f"Minimax: Evaluated {nodes_minimax} nodes in {time_minimax:.4f} seconds")
                print(f"Alpha-Beta: Evaluated {nodes_alpha_beta} nodes in {time_alpha_beta:.4f} seconds")

                # Use Alpha-Beta move (same result, more efficient)
                if move_alpha_beta:
                    row, col = move_alpha_beta
                    self.make_move(row, col, 'O')

            self.current_player = 'O' if self.current_player == 'X' else 'X'

        self.print_board()
        if self.is_winner('X'):
            print("You win!")
        elif self.is_winner('O'):
            print("AI wins!")
        else:
            print("It's a draw!")

def run_test_cases():
    """Runs test cases to demonstrate algorithm functionality."""
    game = TicTacToe()

    # Test case 1: AI can win
    print("\nTest Case 1: AI Winning Position")
    game.board = [
        ['O', 'O', ' '],
        ['X', 'X', ' '],
        [' ', ' ', ' ']
    ]
    move, nodes, time = game.get_best_move_alpha_beta()
    print(f"AI chooses move: {move}, Nodes: {nodes}, Time: {time:.4f}s")

    # Test case 2: Block human win
    print("\nTest Case 2: Block Human Win")
    game.board = [
        ['X', 'X', ' '],
        ['O', 'O', ' '],
        [' ', ' ', ' ']
    ]
    move, nodes, time = game.get_best_move_alpha_beta()
    print(f"AI chooses move: {move}, Nodes: {nodes}, Time: {time:.4f}s")

    # Test case 3: Draw position
    print("\nTest Case 3: Draw Position")
    game.board = [
        ['X', 'O', 'X'],
        ['X', 'O', 'O'],
        ['O', 'X', ' ']
    ]
    move, nodes, time = game.get_best_move_alpha_beta()
    print(f"AI chooses move: {move}, Nodes: {nodes}, Time: {time:.4f}s")

if __name__ == "__main__":
    # Run test cases
    run_test_cases()

    # Play a new game
    game = TicTacToe()
    game.play_game()