import numpy as np
from util import generate_transition

class MDP(object):
    """
    Defines an Markov Decision Process containing:

    - States, S.
    - Actions, A.
    - Rewards, r(s,a).
        *Replaces r(s,a) with f^{a}_{s s'} where f^{a}_{s s'} (x) measures the value of executing action a in state s, moving to state s', and assuming x \in R rewards have already been received.
        * Add mixture operator 'm' where m(p, x, y) = p * x + (1 - p) * y. This is for evaluations from an initial state.
    - Transition Matrix, p(s, a, s').
    - Termination Reward, g(s).

    Includes a set of abstract methods for extended class will need to implement.
    """



    def __init__(self, states=None, actions=None, rewards=None, termination_rewards=None, transitions=None,
                 discount=.99, epsilon=.01):
        """
        :param list states: the valid states in the model.
        :param list actions: the list of actions in the model.
        :param ndarray rewards: reward for each state, action pair.
        :param list termination_rewards: reward for ending in each state.
        :param dict transitions: maps state, action, next state to their probability. transitions[state][action][next_state] = P(next_state | state, action)
        :param float discount: the amount to discount rewards in the next step by.
        :param float epsilon: the accepted noise in value iteration for detecting convergence.
        """   

        # equivalent to S in paper.
        self.s = states
        # equivalent to A in paper.
        self.a = actions
        # Equivalent to r(a, s) in paper.
        self.r = rewards
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
        # Not in paper!
        self.discount = discount
        # Used to check for convergence - minimum change required to perform another iteration.
        self.epsilon = epsilon


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


    def get_reward(self, state, action):
        """
        Gets reward for transition from state->action->nextState.
        :param int state: The current state id.
        :param int action: The current action id.
        :return the reward of that state
        """
        return self.r[state][action]

    
    def getTerminationReward(self, state):
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

        # Initialize V_0 to zero
        values = np.zeros(len(self.s))
        policy = np.zeros(len(self.s), dtype=np.int16)
        options = np.zeros([len(self.s), len(self.a)]) # temp array for easy debugging. Stores value for all

        # Loop until convergence
        # NOTE - Different from paper - paper only runs the number of times as the
        #   horizon.
        while True:
        
            # To be used for convergence check.
            oldValues = np.copy(values)

            # for each state, calculate the action to take based on the valuese
            #   of nearby nodes.
            for i, state in enumerate(self.s):
                # if terminal then just add termination reward for state.
                if self.is_terminal(i):
                    values[i] = self.getTerminationReward(i)
                    continue
                # get value for each action in state.
                for j, action in enumerate(self.a):
                    options[i][j] = self.get_reward(i, j) + self.discount * \
                                    np.dot(self.get_probs(i, j), values)
                # set optimal policy and value.
                policy[i] = np.argmax(options[i])
                values[i] = options[i][policy[i]]

            # Check Convergence.
            if np.max(np.abs(values - oldValues)) <= self.epsilon:
                break
        
        return values, policy


    def policy_iteration(self, policy):
        """
        Evaluate the reward in each state from the policy provided.
        :param ndarray policy: Maps each state to an action to take by
            state index in self.s to action index in state.
        """
        values = np.zeros(len(self.s))

        while True:
            
            # To be used for convergence check.
            oldValues = np.copy(values)

            for i, state in enumerate(self.s):
                # if terminal then just get termination reward for state.
                if self.is_terminal(i):
                    values[i] = self.getTerminationReward(i)
                    continue
                
                act_index = policy[i]
                values[i] = self.get_reward(i, act_index) + self.discount * \
                                np.dot(self.get_probs(i, act_index), values)
                
            # Check Convergence.
            if np.max(np.abs(values - oldValues)) <= self.epsilon:
                break
        
        return values



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

        history = []

        # Run simulation using policy until terminal condition met
        state = self.s.index(start_state)
        reward = 0
        while not self.is_terminal(state):

            act_index = policy[state]
            reward += self.get_reward(state, act_index)
            next_state = self.take_action(state, act_index)
            # track history of simulation. Can be used for reinforcement learning.
            history.append((state, act_index, next_state))

            print(f"In state: {self.s[state]}, taking action: {self.a[act_index]}, moving to state: {self.s[next_state]}")

            state = next_state

        reward += self.getTerminationReward(state)

        return reward, history    



class CoffeeRobot(MDP):

    def __init__(self):
        states = [(a,b) for a in range(1,4) for b in range(1,4)] # (1,1) ... (3,3)
        actions = ['up', 'right', 'down', 'left']
        rewards = generate_rewards(states, actions)
        transitions = generate_transition(states, actions)
        termination = [0] * 8 + [1]

        super().__init__(states, actions, rewards, termination, transitions)
    
    def is_terminal(self, state):
        """
        Checks if this state is a terminal state.
        :param int state: The index of the state to check.
        :return True if terminal, False otherwise.
        """
        if state == 8:
            return True
        return False