import numpy as np
from mdp import MDP
from util import generate_transition

class Paper_MDP(MDP):

    def __init__(self, states=None, actions=None, rewards=None,
                    termination_rewards=None, transitions=None,
                    horizon=None):
        """
        :param list states: the valid states in the model.
        :param list actions: the list of actions in the model.
        :param list rewards: reward for each state.
        :param list termination_rewards: reward for ending in each state.
        :param dict transitions: maps state, action, next state to their probability. transitions[state][action][next_state] = P(next_state | state, action)
        :param int horizon: How many steps in MDP.
        """   

        # equivalent to S in paper.
        self.s = states
        # equivalent to A in paper.
        self.a = actions
        # Equivalent to r(a, s) in paper.
        self.r = rewards
        # Equivalent to g(s) in paper. For our purposes
        self.g = termination_rewards
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
        self.t = np.array(transitions)
        # h in paper.
        self.horizon = horizon


    def value_iteration(self):
        """
        Evaluate the optimal policy.
        :return ndarray values: Value for each state by index.
        :return ndarray policy: Action index for each state by state index.
        """

        # Initialize V_0 to zero
        values = np.zeros(len(self.s))
        policy = np.zeros(len(self.s), dtype=np.int16)
        options = np.zeros([len(self.s), len(self.a)]) # temp array for easy debugging. Stores value for all

        for t in range(self.horizon):

            # for each state, calculate the action to take based on the valuese
            #   of nearby nodes.
            for i, state in enumerate(self.s):
                # if terminal then just add termination reward for state.
                if self.is_terminal(i):
                    values[i] = self.getTerminationReward(i)
                    continue
                # get value for each action in state.
                for j, action in enumerate(self.a):
                    options[i][j] = self.get_reward(i, j) + np.dot(self.get_probs(i, j), values)
                # set optimal policy and value.
                policy[i] = np.argmax(options[i])
                values[i] = options[i][policy[i]]
        
        return values, policy


    def policy_iteration(self, policy):
        """
        Evaluate the reward in each state from the policy provided.
        :param ndarray policy: Maps each state to an action to take by
            state index in self.s to action index in state.
        """
        values = np.zeros(len(self.s))

        for t in range(self.horizon):

            for i, state in enumerate(self.s):
                # if terminal then just get termination reward for state.
                if self.is_terminal(i):
                    values[i] = self.getTerminationReward(i)
                    continue
                
                act_index = policy[i]
                values[i] = self.get_reward(i, act_index) + np.dot(self.get_probs(i, act_index), values)
        
        return values
    

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

            act_index = policy[state]
            reward += self.get_reward(state, act_index)
            next_state = self.take_action(state, act_index)
            # track history of simulation. Can be used for reinforcement learning.
            history.append((state, act_index, next_state))

            print(f"In state: {self.s[state]}, taking action: {self.a[act_index]}, moving to state: {self.s[next_state]}")

            state = next_state
            if self.is_terminal(next_state):
                break

        reward += self.getTerminationReward(state)

        return reward, history


class BasicCoffeeRobot(Paper_MDP):

    def __init__(self):
        states = [(a,b) for a in range(1,4) for b in range(1,4)] # (1,1) ... (3,3)
        actions = ['up', 'right', 'down', 'left']
        rewards = np.array([[-0.01 for _ in range(len(actions))] for _ in range(len(states))])
        transitions = generate_transition(states, actions)
        termination = [0] * 8 + [1]
        horizon = 10

        super().__init__(states, actions, rewards, termination, transitions, horizon)

    def is_terminal(self, state):
        """
        Checks if this state is a terminal state.
        :param int state: The index of the state to check.
        :return True if terminal, False otherwise.
        """
        if state == 8:
            return True
        return False