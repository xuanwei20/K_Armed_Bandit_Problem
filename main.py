import numpy as np

from settings import Settings
from agent import GreedyAgent, EpsilonGreedyAgent
from multi_armed_bandit import MultiArmedBandit
from graph import draw_reward_comparison_graph, draw_action_selection_graph


class ExplorationVsExploitation:
    """Overall class to compare a greedy method with an epsilon-greedy method."""

    def __init__(self, settings):
        """Initialize the comparison."""
        print(f"Implement a {settings.number_of_arms}-armed bandits problem "
              f"with greedy(Ɛ={settings.epsilon_greedy_exploration}) "
              f"and epsilon-greedy(Ɛ={settings.greedy_exploration}) action selection algorithms"
              f"\n")

        n_armed_bandit = MultiArmedBandit(settings.reward_probabilities)  # initialize the multi-armed bandit

        # create two arrays for summing the times when action (1 to k) is selected at each time step for all runs
        # for both greedy and epsilon-greedy methods
        greedy_action_history_sum = np.zeros((settings.num_time_steps, settings.number_of_arms))
        epsilon_action_history_sum = np.zeros((settings.num_time_steps, settings.number_of_arms))

        greedy_reward_history_sum = 0  # initialize the sum of all the reward values a greedy agent receives
        epsilon_reward_history_sum = 0  # initialize the sum of all the reward values an Ɛ-greedy agent receives

        # create a list to conclude all reward history lists of a greedy agent for all runs
        greedy_reward_history_list = []
        # create a list to conclude all reward history lists of an Ɛ-greedy agent for all runs
        epsilon_greedy_reward_history_list = []

        for run in range(settings.num_of_runs):

            # create new agents for each run
            greedy_agent = GreedyAgent(n_armed_bandit)
            epsilon_greedy_agent = EpsilonGreedyAgent(settings.epsilon_greedy_exploration, n_armed_bandit)

            # initialize the sum of reward values a greedy agent receives for one run
            greedy_steps_reward_sum = 0
            # initialize the sum of reward values an Ɛ-greedy agent receives for one run
            epsilon_greedy_steps_reward_sum = 0

            greedy_action_history = []  # create a greedy agent's action history list for one run
            greedy_reward_history = []  # create a greedy agent's reward history list for one run

            epsilon_greedy_action_history = []  # create an Ɛ-greedy agent's action history list for one run
            epsilon_greedy_reward_history = []  # create an Ɛ-greedy agent's reward history list for one run

            for step in range(settings.num_time_steps):

                # list all the actions that a greedy agent takes for all time steps (one run)
                greedy_action_history.append(greedy_agent.action()[0])
                # list all the reward values the greedy agent receives for all time steps (one run)
                greedy_reward_history.append(greedy_agent.action()[1])
                # sum all the reward values for all time steps (one run)
                greedy_steps_reward_sum += greedy_agent.action()[1]

                # list all the actions that an Ɛ-greedy agent takes for all time steps (one run)
                epsilon_greedy_action_history.append(epsilon_greedy_agent.action()[0])
                # list all the reward values the Ɛ-greedy agent receives for all time steps (one run)
                epsilon_greedy_reward_history.append(epsilon_greedy_agent.action()[1])
                # sum all the reward values for all time steps (one run)
                epsilon_greedy_steps_reward_sum += epsilon_greedy_agent.action()[1]

            # the average reward value a greedy agent receives for each time step
            greedy_steps_avg = greedy_steps_reward_sum / settings.num_time_steps
            # the average reward value an Ɛ-greedy agent receives for each time step
            epsilon_greedy_steps_avg = epsilon_greedy_steps_reward_sum / settings.num_time_steps

            print(f"greedy action history: {greedy_action_history}")
            print(f"greedy reward history: {greedy_reward_history}")
            print(f"greedy reward sum: {greedy_steps_reward_sum}")
            print(f"greedy reward average: {greedy_steps_avg}")

            print(f"epsilon-greedy action history: {epsilon_greedy_action_history}")
            print(f"epsilon-greedy reward history: {epsilon_greedy_reward_history}")
            print(f"epsilon-greedy reward sum: {epsilon_greedy_steps_reward_sum}")
            print(f"epsilon-greedy average: {epsilon_greedy_steps_avg}\n")

            # put all the reward-history lists of a greedy agent in one list for all runs
            greedy_reward_history_list.append(greedy_reward_history)
            # put all the reward-history lists of an Ɛ-greedy agent in one list for all runs
            epsilon_greedy_reward_history_list.append(epsilon_greedy_reward_history)

            # sum all the rewards values a greedy agent receives for all runs
            greedy_reward_history_sum += greedy_steps_reward_sum
            # sum all the rewards value an Ɛ-greedy agent receives for all runs
            epsilon_reward_history_sum += epsilon_greedy_steps_reward_sum

            # summing action history for both greedy and epsilon-greedy methods
            greedy_action_history = np.array(greedy_action_history)
            for i, action in enumerate(greedy_action_history):
                greedy_action_history_sum[i][action-1] += 1
            epsilon_greedy_action_history = np.array(epsilon_greedy_action_history)
            for i, action in enumerate(epsilon_greedy_action_history):
                epsilon_action_history_sum[i][action-1] += 1

        print(f"Greedy action history sum:\n "
              f"{greedy_action_history_sum}\n")
        print(f"Epsilon-greedy action history sum:\n "
              f"{epsilon_action_history_sum}\n")

        # average the greedy agent's reward values of all runs for every time step
        greedy_reward_average_arrays = [np.array(x) for x in greedy_reward_history_list]
        greedy_reward_average_list = [np.mean(k) for k in zip(*greedy_reward_average_arrays)]
        print(f"greedy reward average list: {greedy_reward_average_list}")
        # average the Ɛ-greedy agent's reward values of all runs for every time step
        epsilon_greedy_reward_average_arrays = [np.array(x) for x in epsilon_greedy_reward_history_list]
        epsilon_greedy_reward_average_list = [np.mean(k) for k in zip(*epsilon_greedy_reward_average_arrays)]
        print(f"epsilon-greedy reward average list: {epsilon_greedy_reward_average_list}\n")

        # calculate the overall average reward value for both agents
        greedy_reward_history_average = greedy_reward_history_sum / (settings.num_time_steps * settings.num_of_runs)
        epsilon_reward_history_average = epsilon_reward_history_sum / (settings.num_time_steps * settings.num_of_runs)
        print(f"Result after {settings.num_of_runs} runs with {settings.num_time_steps} time steps: \n"
              f"Greedy method average reward: {greedy_reward_history_average} \n"
              f"Epsilon-greedy method average reward: {epsilon_reward_history_average}\n")

        # x axis for all three graphs
        x_axis = range(1, settings.num_time_steps + 1)

        # draw a graph to compare the averaged reward values for every time steps
        list_1 = greedy_reward_average_list
        list_2 = epsilon_greedy_reward_average_list
        epsilon = settings.epsilon_greedy_exploration
        draw_reward_comparison_graph(x_axis, list_1, list_2, epsilon)

        # parameters for both greedy and epsilon-greedy action probability graphs
        number_of_arms = settings.number_of_arms
        num_runs = settings.num_of_runs

        # draw a graph to check action selection percentage for greedy
        action_selection_history_sum = greedy_action_history_sum
        epsilon = settings.greedy_exploration
        draw_action_selection_graph(x_axis, epsilon, number_of_arms, num_runs, action_selection_history_sum)

        # draw a graph to to check action selection percentage for epsilon-greedy
        epsilon = settings.epsilon_greedy_exploration
        action_selection_history_sum = epsilon_action_history_sum
        draw_action_selection_graph(x_axis, epsilon, number_of_arms, num_runs, action_selection_history_sum)


if __name__ == "__main__":
    # start simulation
    ExplorationVsExploitation(Settings)
