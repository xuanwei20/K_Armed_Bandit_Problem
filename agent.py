import random


class Agent:
    """A class to manage an intelligent agent."""

    def __init__(self, exploration_percentage, multi_armed_bandit):
        """Initialize an agent."""
        self.multi_armed_bandit = multi_armed_bandit
        self.number_of_arms = multi_armed_bandit.get_number_of_arms()
        self.n = [0] * self.number_of_arms  # initialize N(a)<-0, number of selecting an action (for action 1 to k)
        self.q = [0.0] * self.number_of_arms  # initialize Q(a)<-0, estimated value of each action (for action 1 to k)
        self.exploration_percentage = exploration_percentage

    def update_reward(self, arm_index, reward):
        """Update the times(n) that an action is selected and its corresponding estimated value(q)."""
        self.n[arm_index] += 1  # update that number of action which has been selected
        self.q[arm_index] += (1 / self.n[arm_index]) * (reward - self.q[arm_index])  # update estimated value
        return self.n

    def policy(self):
        """The learning agent's way of behaving."""
        random_selection = random.random()

        # exploration or exploitation based on the exploration percentage (epsilon value)
        if random_selection < self.exploration_percentage:  # random choice from all the actions
            random_arm = random.choice(range(self.number_of_arms))
            return random_arm

        else:
            maximum_reward = max(self.q)  # find highest estimated value
            maximum_reward_arms = []  # select action with the highest estimated value
            for index, value in enumerate(self.q):
                if value == maximum_reward:
                    maximum_reward_arms.append(index)
            best_arm = random.choice(maximum_reward_arms)
            return best_arm

    def action(self):
        """Take an action on which arm to chose."""
        chosen_arm = self.policy()
        reward = self.multi_armed_bandit.pull_arm(chosen_arm)
        self.update_reward(chosen_arm, reward)
        return chosen_arm+1, reward


class GreedyAgent(Agent):
    """Represent aspects of an intelligent agent, specific to greedy agent."""

    def __init__(self, multi_armed_bandit):
        """Initialize attributes of the greedy agent."""
        exploration_percentage = 0
        super().__init__(exploration_percentage, multi_armed_bandit)


class EpsilonGreedyAgent(Agent):
    """Represent aspects of an intelligent agent, specific to epsilon greedy agent."""

    def __init__(self, exploration_percentage, multi_armed_bandit):
        """Initialize attributes of the epsilon greedy agent."""
        super().__init__(exploration_percentage, multi_armed_bandit)
