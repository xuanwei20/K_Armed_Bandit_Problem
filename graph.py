import matplotlib.pyplot as plt


def draw_reward_comparison_graph(x_axis, list_1, list_2, epsilon):
    """A graph to show average performances of greedy and epsilon-greedy methods."""
    plt.clf()
    plt.title(f"Greedy Method(Ɛ=0) vs Epsilon-Greedy Method(Ɛ={epsilon})")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.xlim([0, len(x_axis)])
    plt.ylim([0, 1])
    plt.plot(x_axis, list_1, label="Ɛ=0")
    plt.plot(x_axis, list_2, label=f"Ɛ={epsilon}")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.show()


def draw_action_selection_graph(x_axis, epsilon, number_of_arms, num_runs, action_selection_history_sum):
    """A graph to show probabilities of selecting every action."""
    plt.clf()
    for k in range(number_of_arms):
        greedy_action_history_sum_plot = action_selection_history_sum[:, k]/num_runs
        plt.plot(x_axis, greedy_action_history_sum_plot, label=f"arm {k+1}")

    plt.title(f"Probability of Selecting Actions(Ɛ={epsilon})")
    plt.xlabel("Time Steps")
    plt.ylabel("Action Selection Probability")
    plt.xlim([0, len(x_axis)])
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.show()
