import numpy as np

def generate_transition(states, actions, terminals=[]):
        act_map = {
                'up' : lambda state: (state[0] + 1, state[1]),
                'down': lambda state: (state[0] - 1, state[1]),
                'right': lambda state: (state[0], state[1] + 1),
                'left': lambda state: (state[0], state[1] - 1)
                }
        # Maps an action to the sequences of actions that may actually be taken and the probability of
        #    that sequence of actions occuring. Not sure if this is generic enough to be in the main implementation,
        #    but it is good for generating them in this case.
        probs = {
                'up': {('up',): 0.9, ('up', 'left'): 0.1 },
                'right': {('right',): 0.9, ('right', 'up') : 0.1},
                'down': {('down',): 0.9, ('down', 'right'): 0.1},
                'left': {('left',): 0.9, ('left', 'down'): 0.1}
                }

        # Generate transition matrix from these rules.
        transition = np.zeros((len(states), len(actions), len(states)))
        for state in states:
            for action in actions:
                # for each ACTUAL action sequence that results from attempting to take this action.
                for sequence, prob in probs[action].items():
                    next_state = state
                    # apply each action in the sequence.
                    for sub_action in sequence:
                        temp = act_map[sub_action](next_state)
                        if not temp in states:
                            # This sequence of actions resulted in leaving the grid. Lets assume it stops here instead.
                            break
                        next_state = temp
                    # Get the states indexes.
                    s = states.index(state)
                    a = actions.index(action)
                    sp = states.index(next_state)
                    transition[s][a][sp] += prob
        return transition



def generate_rewards(states, actions, value):
    return np.full((len(states), len(actions), len(states)), -0.01)
