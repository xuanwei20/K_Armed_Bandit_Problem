class Settings:
    """A class to store all settings for a multi-armed bandit problem."""

    reward_probabilities = [0.1, 0.3, 0.7, 0.7, 0.5]  # reward probabilities of the multi-armed bandit
    number_of_arms = len(reward_probabilities)
    num_time_steps = 100  # number of action selections
    num_of_runs = 200  # number of repeating the {num_time_steps} action selections
    greedy_exploration = 0  # probability of exploration in greedy method
    epsilon_greedy_exploration = 0.4  # probability of selecting randomly from among all the actions
