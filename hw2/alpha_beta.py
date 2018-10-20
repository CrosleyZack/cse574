#!/usr/bin/env python

import sys
import math
from operator import itemgetter
from time import time


class State:
    def children(self):
        '''Returns an iterator over child states.
        NOTE: Valid only if NOT is_terminal().
        '''
        raise NotImplementedError()

    def payoff(self):
        '''Returns the payoff of the state.
        NOTE: Valid only if is_terminal().
        '''
        raise NotImplementedError()

    def payoff_lower(self):
        '''Returns a lower bound on the payoff.'''
        raise NotImplementedError()

    def payoff_upper(self):
        '''Returns an upper bound on the payoff.'''
        raise NotImplementedError()

    def is_terminal(self):
        '''Checks if the state is terminal.'''
        raise NotImplementedError()

    def is_max_player(self):
        '''Checks if the current state is a max player's turn.'''
        raise NotImplementedError()


class TicTacToe(State):
    def __init__(self, board, player, is_max_player, move=None):
        self._board = board
        self._player = player
        self._is_max_player = is_max_player
        self._move = move

    def children(self):
        player = 'X' if self._player == 'O' else 'O'
        is_max_player = not self._is_max_player # the child is the opposite type of player as its parent (zero sum).

        # For each empty space...
        for r in range(3):
            for c in range(3):
                if self._board[r][c] == '_':
                    board = [[x for x in row] for row in self._board] # Get copy of board.
                    board[r][c] = self._player # set r, c to this players token. 
                    yield TicTacToe(board, player, is_max_player, (r, c)) # create new board.

    def payoff(self):
        winner = self._winner()

        if winner is None:
            return 0

        # Either previous min-player won (-1) or previous max-player won (+1).
        return -1 if self._is_max_player else 1

    def payoff_lower(self):
        return -1

    def payoff_upper(self):
        return 1

    def is_terminal(self):
        # if someone has won, it's terminal.
        if self._winner() is not None:
            return True

        # If there are still empty spaces, it's not terminal.
        for r in range(3):
            for c in range(3):
                if self._board[r][c] == '_':
                    return False
        # If there are no empty spaces, it's terminal.
        return True

    def is_max_player(self):
        return self._is_max_player

    def move(self):
        '''Returns the move used to transition to this state.'''
        return self._move

    def _winner(self):
        '''Returns the current winner, if one exists.'''
        board = self._board

        for i in range(3):
            # Check rows...
            if board[i][0] != '_' and \
               board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]

            # Check columns...
            if board[0][i] != '_' and \
               board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]

        # Check diagonals...
        if board[0][0] != '_' and \
           board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]

        if board[2][0] != '_' and \
           board[2][0] == board[1][1] == board[0][2]:
            return board[2][0]

        return None
    
    # get string representation of state.
    def __str__(self):
        for row in self._board:
            print(str(row[0]) + " " + str(row[1]) + str(row[2]))
        print("Player " + self._player + ", Move " + str(self._move))


def alpha_beta_minimax(state):

    def evaluate_ab(state, alpha=float('-inf'), beta=float('inf'), depth=1):
        if state.is_terminal():
            return state.payoff()
        if state.is_max_player():
            best_val = alpha
            for child in state.children():
                best_val = max(best_val, evaluate_ab(child, best_val, beta, depth+1))
                if beta <= best_val:
                    break
        else:
            best_val = beta
            for child in state.children():
                best_val = min(best_val, evaluate_ab(child, alpha, best_val, depth+1))
                if best_val <= alpha:
                    break
        return best_val


    optimal = (float('-inf'), None)
    for child in state.children():
        value = evaluate_ab(child)
        if optimal[0] < value:
            optimal = (value, child)
    return optimal[1]


# NOTE currently always starts with maximize.
def minimax(state):

    # Gets the minimax value of this state.
    def evaluate(state):
        if state.is_terminal():
            return state.payoff()
        func = max if state.is_max_player() else min
        return func(evaluate(child) for child in state.children())

    optimal = (float('-inf'), None)
    for child in state.children():
        value = evaluate(child)
        if optimal[0] < value:
            optimal = (value, child)
    return optimal[1]

def main():
    player = raw_input() # Get which player this is. Should be 'X' or 'O'
    board = [list(raw_input()) for _ in range(3)] # get three inputs from player describing current board state. Unused spaces should be encoded as '_'
    # board = [['_', '_', 'X'], ['_', 'O', '_'], ['X', 'O', '_']] # shortcut for testing. initialize to default.
    # Checks for valid board / player values. Prevent running down bugs from invalid input.
    if not player in ['X', 'O']:
        raise ValueError('The entered value was not a valid player token: ' + str(player))
    if not len(board) == 3 or not all([len(row) == 3 for row in board]) or not all([char in ['_', 'X', 'O'] for row in board for char in row]):
        raise ValueError('The board entered was not valid: ' + str(board))
    state = TicTacToe(board, player, True)
    state = alpha_beta_minimax(state)
    print('%d %d' % state.move())



checked_states = 0
if __name__ == '__main__':
    main()
