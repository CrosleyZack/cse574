import numpy as np
from util import generate_rewards, generate_transition

class FRMDP(object):
    """
    Describes the Functional Reward MDP described in "Markov Decision Processes with
    Functional Rewards" by Olivier Spanjaard and Paul Weng.

    Based loosely on the implementation of a standard MDP by Joey Velez-Ginorio at:
    http://pythonfiddle.com/markov-decision-process/
    """

    def __init__(self, states=None, actions=None, termination_rewards=None, transitions=None,
                       horizon=None):
        """
        :param list states: the valid states in the model.

        :param list actions: the list of actions in the model.
        :param list termination_rewards: reward for ending in each state.
        :param ndarray transitions: maps state, action, next state to their probability. transitions[state][action][next_state] = P(next_state | state, action)
        :param int horizon: number of rounds in the decision process.
        """
        # equivalent to S in paper.
        self.s = states
        # equivalent to A in paper.
        self.a = actions
        # Equivalent to g(s) in paper. For our purposes
        self.g = termination_rewards
        # Equivalent to p in paper.
        #
        # Currently a 3 dim matrix, where [state][action][s'] = prob of s - action - > s'.
        # getTransitionStatesAndProbs(i) returns 2d matrix where:
        #      action -> [ next states ] -> probability.
        #
        #                 | State 1    | State 2    | ...
        #     +-----------+------------+------------+-------
        #     | Action 1  | P(S1 | A1) | P(S2 | A1) | ...
        #     +-----------+------------+------------+-------
        #     | Action 2  | P(S1 | A2) | P(S2 | A2) | ...
        #     +-----------+------------+------------+-------
        #     | Action 3  | P(S1 | A3) | P(S2 | A3) | ...
        #     +-----------+------------+------------+-------
        #     | ...       |  ...       | ...        | ...
        #
        #
        #      This implies each action has a full list of states for next-states and many may
        #      simply have 0 probability. Thus a dot product w/ self.values will perform the
        #      full P(s') * val(s') operation.
        self.t = transitions
        # h in paper.
        self.horizon = horizon

    def is_terminal(self, state):
        """
        Checks if MDP is in terminal state.
        :param int state: The index of state to check if is terminal.
        :return True if the state is terminal, False otherwise.
        """
        raise NotImplementedError()

    def get_probs(self, state, action):
        """
        Returns the list of transition probabilities.
        :param int state: The state index to transition out of.
        :param int action: The index of action to take in state. If none, returns for all vectors in state.
        :return the vector  of probabilities for each next state given an action and state, or a matrix of probabilities for each action.
        :rtype ndarray.
        """
        return self.t[state][action][:]

    def get_reward(self, state, action, state_reward):
        """
        Gets reward for transition from state->action->nextState.
        :param int state: The current state id.
        :param int action: The current action id.
        :param ndarray state_reward: The vector of rewards from the previous iteration of this state, action pair.
        :return vector of rewards for each next_state.
        """
        raise NotImplementedError()

    def termination_reward(self, state):
        """
        Checks if MDP is in terminal state.
        :param int state: The index of state to check if is terminal.
        :return termination reward for this state.
        """
        return self.g[state]

    def take_action(self, state, action):
        """
        Take an action in an MDP, return the next state. Chooses according to probability distribution of state transitions, contingent on actions. Works by using using array of states and companion array of probabilities for that next state given this state and action and computing which occurs at random.

        :param state state: starting state to take action from.
        :param action action: action to take in state.
        :return next state by random selection from weighted distribution.
        """
        return np.random.choice(range(len(self.s)), p=self.get_probs(state, action))

    def value_iteration(self):
        """
        Evaluate the optimal policy.
        :return ndarray values: Value for each state by index.
        :return ndarray policy: Action index for each state by state index.
        """
        policy = np.zeros(len(self.s), dtype=np.int16)
        # q(s,a) in the paper.
        options = np.zeros([len(self.s), len(self.a)])
        # delta in the paper. Stores the value from the previous iteration.
        delta = np.zeros([self.horizon + 1, len(self.s)])
        delta[0] = self.g # initialize delta[0] to the terminal values.

        for t in range(1, self.horizon + 1):
            # for each state, calculate the action to take based on the valuese
            #   of nearby nodes.
            for s, state in enumerate(self.s):
                # if this is a terminal state, skip the computation and just set it to the terminal value.
                if self.is_terminal(s):
                    delta[t][s] = delta[t-1][s]
                    continue
                # get value for each action in state.
                for a, action in enumerate(self.a):
                    options[s][a] = np.dot(self.get_probs(s, a),
                                           self.get_reward(s, a, delta[t - 1]))
                # set optimal policy and value.
                policy[s] = np.argmax(options[s])
                delta[t][s] = options[s][policy[s]]

        return delta[self.horizon], policy


    def policy_iteration(self, policy):
        """
        Evaluate the reward in each state from the policy provided.
        :param ndarray policy: Maps each state to an action to take by
            state index in self.s to action index in state.
        """

        # delta in the paper. Stores the value from the previous iteration.
        delta = np.zeros([self.horizon + 1, len(self.s)])
        delta[0] = self.g

        for t in range(1, self.horizon + 1):
            # for each state, calculate the action to take based on the valuese
            #   of nearby nodes.
            for s, state in enumerate(self.s):
                # if this is a terminal state, skip the computation and just set it to the terminal value.
                if self.is_terminal(s):
                    delta[t][s] = delta[t-1][s]
                    continue
                a = policy[s]
                delta[t][s] = np.dot(self.get_probs(s, a),
                                     self.get_reward(s, a, delta[t - 1]))

        return delta[self.horizon]


    def describe_policy(self, policy):
        """
        Create a textual description for a human user of the policy.
        :param ndarray policy: Maps each state to an action to take by
            state index in self.s to action index in state.
        """
        for s, state in enumerate(self.s):
            act_index = policy[s]
            action = self.a[act_index]
            print(f"Perform action {action} in state {state}.")


    def simulate(self, start_state, policy):
        """
        Runs a single iteration of the policy provided from the given state
            and returns the resulting reward.
        :param state start_state: State from which to begin simulation.
        :param ndarray policy: Maps each state to an action to take by
            state index in self.s to action index in state.a
        """
        # Run simulation using policy until terminal condition met
        state = self.s.index(start_state)
        history = []
        reward = 0
        for t in range(self.horizon):
            # Get the stochastic next_state and reward.
            act_index = policy[state]
            next_state = self.take_action(state, act_index)
            reward += self.get_reward(state, act_index, next_state)
            # Track history of simulation. Can be used for reinforcement learning.
            history.append((state, act_index, next_state))
            # Print transition.
            print(f"In state: {self.s[state]}, taking action: {self.a[act_index]}, moving to state: {self.s[next_state]}")
            # Repeat if next state isn't terminal.
            state = next_state
            if self.is_terminal(next_state):
                break

        reward += self.termination_reward(state)
        return reward, history



class MDP(FRMDP):
    """
    FRMDP implementation of a basic MDP. This has been validated by comparing to a known
    correct MDP implementation.
    """

    def __init__(self, states=None, actions=None, termination_rewards=None, transitions=None,
                       horizon=None, rewards=None, discount=1):
        """
        :param list states: the valid states in the model.
        :param list actions: the list of actions in the model.
        :param list termination_rewards: reward for ending in each state.
        :param ndarray transitions: maps state, action, next state to their probability. transitions[state][action][next_state] = P(next_state | state, action)
        :param int horizon: number of rounds in the decision process.
        :param ndarray rewards: maps state, action, next state to their reward.
        :param float discount: The ratio indicating importance of future rewards. By default is one, meaning future rewards are as important as current rewards.
        """
        self.rewards = rewards
        self.discount = discount
        super().__init__(states, actions, termination_rewards, transitions, horizon)

    def get_reward(self, state, action, state_reward):
        """
        Gets reward for transition from state->action->nextState.
        :param int state: The current state id.
        :param int action: The current action id.
        :param ndarray state_reward: The vector of rewards from the previous iteration of this state, action pair.
        :return vector of rewards for each next_state.
        """
        return self.rewards[state][action] + (self.discount * state_reward)



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