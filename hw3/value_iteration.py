##
# python script to try to give the value iteration values
#   more easily
# Assumes a 2d array.
from numpy import array
from operator import itemgetter
from itertools import product
from sys import argv


def is_valid(state, board):
    for index, size in zip(state, board.shape):
        if index < 0 or index >= size:
            return False
    return True


def valid_states(board):
    states = []
    x, y = board.shape
    for x_val, y_val in product(range(x), range(y)):
        states.append((x_val, y_val))
    return states


def print_board(board):
    print('\n')
    print(str(board))
    print('\n')


def row_iterator(board):
    x, y = board.shape
    for i in range(y):
        row = []
        for j in range(x):
            row.append((j,i))
        yield row


def state_iterator(board):
    x, y = board.shape
    for i in range(y):
        for j in range(x):
            yield(x,y)


def get_neighbors(state, actions, board):

    neighbors = []
    for name, action in actions.items():
        new_state = action(state)
        if is_valid(new_state, board):
            neighbors.append(new_state)
    return neighbors


##########################################


def init(R = 100):
    board = array([[0,0,0], [0,0,0], [R, 0, 10]])
    goal = (2,2)
    actions = {
            'up': lambda state: (state[0] + 1, state[1]),
            'right': lambda state: (state[0], state[1] +1 ),
            'down': lambda state: (state[0] - 1, state[1]),
            'left': lambda state: (state[0], state[1] - 1)
          }
    probs =   {
            'up': {'up': 0.8, 'left': 0.1, 'right': 0.1},
            'right': {'right': 0.8, 'up': 0.1, 'down': 0.1},
            'down': {'down': 0.8, 'left': 0.1, 'right': 0.1},
            'left': {'left': 0.8, 'up': 0.1, 'down': 0.1}
          }
    gamma = 0.99
    cost = -0.04
    return board, goal, actions, probs, gamma, cost



def get_value_iteration(board, goal, actions, probs,  gamma, cost, iterations = 1):

    for i in range(iterations):

        rewards = board.copy()

        print("BOARD")
        print_board(board)
        print("REWARDS")
        print_board(rewards)
        print("Iteration " + str(i) + ':')

        state_to_action = {}
        calculated = [ goal ]
        to_do = [ neighbor for neighbor in get_neighbors(goal, actions, board) ]

        while to_do:

            node = to_do.pop()
            action_dict = get_values(node, board, rewards, actions, probs, gamma, cost)
            state_to_action[node] = action_dict

            # add state to completed
            calculated.append(node)
            neighbors = get_neighbors(node, actions, board)
            to_do.extend([neighbor for neighbor in neighbors
                          if not neighbor in to_do and not neighbor in calculated])

        # print out the decisions
        # print_latex(state_to_action)
        # print_state_to_action(state_to_action, max_only=True)
        print_actions(state_to_action, board)




def get_value(state, board, action, probs, gamma, cost):
    value_under_policy = 0
    # for each s'
    for actual_action, prob in probs[action].items():
        # Get the next state if this action is taken.
        s_prime = actions[actual_action](state)
        if not is_valid(s_prime, board):
            s_prime = state

        contrib = prob * (




def get_values(state, board, rewards, actions, probs, gamma, cost, update_board = True):
    """
    Gets the values for each possible action in the function.
    :return  dictionary mapping action to value
    """

    # Remember value == v^{*} (s) = R(s) + \gamma \max_a \sum_{s'} p(s,a,s') v^{*} (s')
    # This appears to be wrong - look up actual formula.

    action_value_dict = {}
    for name, action in actions.items():
        value = 0
        # for each possible action that can result from taking this action
        for actual_action, prob in probs[name].items():
            next_state = actions[actual_action](state)
            if not is_valid(next_state, board):
                # then we will calculate for staying put.
                this_value = prob * rewards[state]
                value += this_value
            else:
                # calculate this term.
                this_value = prob * rewards[next_state]
                value += this_value

        action_value_dict[name] = rewards[state] + gamma * value

    # Update board value at state
    if update_board:
        action, value = max(action_value_dict.items(), key=itemgetter(1))
        board[state] = value

    return action_value_dict



######################################################
## I/O

def print_state_to_action(state_to_action, max_only = False):
    """
    Print the state to action dictionary.
    :param state_to_action dictionary mapping each state (table node (i,j)) to a dictionary of possible actions to values.
    """
    for node, act_dict in state_to_action.items():

        print('\n\tNode ' + str(node) + ':')

        if not max_only:
            for action, value in act_dict.items():
                print('\t\tAction: ' + str(action) + ',\tValue: ' + str(value))

        max_action = [action for action, value in act_dict.items() if value == max(act_dict.values())][0]
        print ('\t\tMax Action: ' + str(max_action) + ', Max Value: ' + str(act_dict[max_action]))


def print_actions(state_to_action, board):
    print("for debug...")
    print(str(state_to_action.keys()))

    for row in row_iterator(board):
        to_print = ''
        for node  in row:
            if node == (2,2):
                to_print += '.  '
                continue
            act_dict = state_to_action[node]
            max_action = [action for action, value in act_dict.items() if value == max(act_dict.values())][0]
            to_print += max_action + '  '
        print(to_print)
    print('\n\n')


def print_latex(state_to_action, iteration=1):

    keys = list(state_to_action.keys())
    keys.sort()
    print('\\begin{{large}}Iteration ' + str(iteration) + '\\end{{large}}')
    print('\\begin{{itemize}}')
    for state in keys:
        options = state_to_action[state] # this is the map from action to value.
        print('\\item[State ' + str(state) + ']')
        #PRINT BASIC EQUATION
        print("\\begin{{equation}}v(" + str(state) + ") = \max_a \sum_{{s'}} p(s'|" + str(state) + ",action)\\left[R(" + str(state) + ")+" + str(gamma) + "v(s')\\right]\\end{{equation}}")
        #GET EQUATION FOR THIS STATE AND EACH INTENDED ACTION
        eq = "\\begin{{equation}}v(" + str(state) + ") = \max_a \\left("
        for i, (action, value) in enumerate(options.items()):
            if not i == 0:
                eq += ',\\:'
            eq += '\\left['
            # GET ITEMS TO TAKE SUM OF
            for j, (actual_action, prob) in enumerate(probs[action].items()):
                next_state = actions[actual_action](state)
                if not is_valid(next_state, board):
                    next_state = state
                if not j == 0:
                    eq += '\\plus'
                eq += str(prob) + '\\left[' + str(rewards[state]) + '\\plus' + str(gamma) + '\\times' + str(rewards[next_state]) + '\\right]'
            eq + '\\right]'
        eq += '\\right)\\end{{equation}}'
        print(eq)
        # SHOW VALUES TO MAX BETWEEN
        eq = "\\begin{{equation}}v(" + str(state) + ") = \max_a \\left("
        for i, (action, value) in enumerate(options.items()):
            if not i == 0:
                eq += ',\\:'
            eq += str(value)
        eq += '\\right)\\end{{equation}}'
        print(eq)
        # SHOW FINAL VALUE
        print("\\begin{{equation}}v(" + str(state) + ") = " + str(max(options.values())) + '\\end{{equation}}')
    print('\\end{{itemize}}')


if __name__ == '__main__':
    if len(argv) < 2:
        r = 100
    else:
        r = int(argv[1])
    board, goal, actions, rewards, gamma, cost = init(r)
    get_value_iteration(board, goal, actions, rewards, gamma,  cost, iterations=2)
