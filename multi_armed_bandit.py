from random import random


class MultiArmedBandit:
    """A class to manage a multi-armed bandit."""

    def __init__(self, reward_probabilities):
        """Initialize the multi-armed bandit."""
        self.reward_probabilities = reward_probabilities  # reward probabilities of every arm

    def get_number_of_arms(self):
        return len(self.reward_probabilities)

    def pull_arm(self, arm_index):
        """The multi-armed bandit gives reward after every action selection."""
        random_reward_indicator = random()  # Binomial distribution, random float: 0.0 <= x <1.0
        if random_reward_indicator < self.reward_probabilities[arm_index]:
            reward = 1
        else:
            reward = 0
        return reward
