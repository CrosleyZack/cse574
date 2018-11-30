from frmdp import MDP, FRMDP
import numpy as np

class BasicCoffeeRobot(MDP):
    """
    MDP for value iteration including spills cost.
    """

    def __init__(self):
        states = [(a,b) for a in range(1,4) for b in range(1,4)] # (1,1) ... (3,3)
        actions = ['up', 'right', 'down', 'left']
        rewards = generate_rewards(states, actions, -0.01)
        transitions = generate_transition(states, actions)
        termination = [0] * 8 + [1]
        horizon = 10
        super().__init__(states, actions, termination, transitions, horizon, rewards)


    def get_reward(self, state, action, state_reward):
        """
        Gets reward for transition from state->action->nextState.
        :param int state: The current state id.
        :param int action: The current action id.
        :param ndarray state_reward: The vector of rewards from the previous iteration of this state, action pair.
        :return vector of rewards for each next_state.
        """
        # to_subtract =  0.1 * (np.full(len(self.s), -1 * state_reward))
        p_spill = 0.03
        return (1 - p_spill) * (self.rewards[state][action] + (self.discount * state_reward)) + p_spill * (np.full(len(self.s), -1))

    def is_terminal(self, state):
        """
        Checks if this state is a terminal state.
        :param int state: The index of the state to check.
        :return True if terminal, False otherwise.
        """
        if state == 8:
            return True
        return False



class CoffeeRobot(FRMDP):
    """
    Coffee Robot example from the paper. A robot on a 3x3 grid needs to deliver coffee
    to the space (3,3). The robot can move in any of four directions, but may end up
    to the left of the intended space due to a fault in the code. The cost of spilling
    in a space is:
                        (3,3,)
        +----+----+----+
        | -1 | -2 | 0  |
        +----+----+----+
        | -3 | -3 | -4 |
        +----+----+----+
        | -1 | -2 | -1 |
        +----+----+----+
    (1,1)

    The probability of spilling the coffee is characterized by the following matrix.
                        (3,3,)
        +----+----+----+
        | 1% | 5% | 3% |
        +----+----+----+
        | 2% | 4% | 2% |
        +----+----+----+
        | 3% | 5% | 3% |
        +----+----+----+
    (1,1)

    This could not be readily implemented in a standard MDP, however in a functional MDP
    it is fairly straight forward.

    NOTE At value_iteration t=4 the policy matches the output in the paper.
    """

    def __init__(self):
        """
        Initialize the values for the coffee robot example in the paper.
        """
        states = [(a,b) for a in range(1,4) for b in range(1,4)]
        actions = ['up', 'right', 'down', 'left']
        transitions = generate_transition(states, actions)
        termination = [0] * 8 + [1]
        horizon = 100
        self.spill_cost = np.array([-1,-2,-1, -3,-3,-4, -1,-2,0])
        self.spill_prob = np.array([0.03,0.05,0.03, 0.02,0.04,0.02, 0.01,0.05,0.03])
        super().__init__(states, actions, termination, transitions, horizon)


    def get_reward(self, state, action, state_reward):
        """
        Gets reward for transition from state->action->nextState.
        :param int state: The current state id.
        :param int action: The current action id.
        :param ndarray state_reward: The vector of rewards from the previous iteration of this state, action pair.
        :return vector of rewards for each next_state.
        """
        # P * r + (1-P) * x.
        return self.spill_prob * self.spill_cost + (1 - self.spill_prob) * state_reward

    def is_terminal(self, state):
        """
        Checks if this state is a terminal state.
        :param int state: The index of the state to check.
        :return True if terminal, False otherwise.
        """
        if state == 8:
            return True
        return False


def generate_transition(states, actions, terminals=[], p_success = 1):
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
                    transition[s][a][sp] += p_success * prob
        return transition


def generate_rewards(states, actions, value):
    return np.full((len(states), len(actions), len(states)), -0.01)


def main():
    # Create basic coffee robot, perform value iteration, and ensure the 
    #   values match the policy iteration of that optimal policy.
    coffee = BasicCoffeeRobot()
    opt_val1, opt_policy1 = coffee.value_iteration()
    test_val1 = coffee.policy_iteration(opt_policy1)
    assert(np.array_equal(opt_val1, test_val1))
    # Create coffee robot, perform value iteration, and ensure the 
    #   values match the policy iteration of that optimal policy.
    coffee2 = CoffeeRobot()
    opt_val2, opt_policy2 = coffee2.value_iteration()
    test_val2 = coffee2.policy_iteration(opt_policy2)
    assert(np.array_equal(opt_val2, test_val2))
    # Assert they produce different policies and value outputs, as one would expect.
    assert(not np.array_equal(opt_val1, opt_val2))
    assert(not np.array_equal(opt_policy1, opt_policy2))
    # Display derived policies.
    print("\nMDP Coffee Robot Policy.")
    coffee.describe_policy(opt_policy1)
    print("-"*50) # separator.
    print("\nFRMDP Coffee Robot Policy.")
    coffee.describe_policy(opt_policy2)


if __name__ == "__main__":
    main()