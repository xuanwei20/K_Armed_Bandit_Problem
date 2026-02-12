# 📊 K-Armed Bandit Problem

A Python implementation of the **K-armed bandit problem** — a classic reinforcement learning problem that demonstrates the trade-off between **exploration and exploitation**. This repository includes implementations of agents that use **greedy** and **ε-greedy** strategies to select actions and compare their performance.

---

## 🔍 Overview

In the **k-armed bandit problem**, an agent must repeatedly choose from **k actions (arms)**, each providing a reward drawn from an unknown probability distribution. The goal is to **maximize cumulative rewards** over time by learning which actions are better. :contentReference[oaicite:0]{index=0}

This project implements a basic simulation of this problem with simple agent strategies:

- 🟢 **Greedy** – always chooses the best known action  
- 🔁 **ε-Greedy** – explores randomly with probability ε, otherwise chooses greedily  

---

## 📁 Repository Structure

```
K_Armed_Bandit_Problem/
│
├── main.py                  # Entry point to run the simulation
├── multi_armed_bandit.py    # Bandit environment and reward logic
├── agent.py                 # Agent classes (Greedy/Epsilon-Greedy)
├── graph.py                 # Plotting and performance visualization
├── settings.py              # Configuration (k arms, steps, epsilon, etc.)
├── LICENSE
├── README.md                # This file
└── .gitignore
```

---

## 🧠 How It Works

1. **Settings** define the number of arms, steps, and algorithm parameters.
2. `main.py` initializes:
   - A **MultiArmedBandit** environment
   - One or more agent instances
3. The agents interact with the environment by selecting actions and receiving rewards.
4. Rewards and results (e.g., average reward over time) can be visualized with the plotting functions.

This structure separates concerns — environment, agent logic, settings, and visualization — for clarity and extensibility.

Description of the Code

Multi-armed bandit problem's main module starts a class called ExplorationVsExploitation. Settings is imported into the main program file; MultiArmedBandit is imported to make an instance of MultiArmedBandit; GreedyAgent, EpsilonGreedyAgent are imported to make an instance of GreedyAgent and an instance of EpsilonGreedyAgent; draw_graph function is imported from graph module.

The settings module contains a class called Settings to store all the values in one place instead of adding settings throughout the code. Simply changing some values in settings.py can modify the multi-armed bandit problem.

The multi_armed_bandit module contains the class MultiArmedBandit.

---

## 🛠️ Requirements

- Python 3.x

No external dependencies are required unless optional plotting (e.g., `matplotlib`) is used.

---

## ▶️ Running the Project

1. **Clone the repository**

```bash
git clone https://github.com/xuanwei20/K_Armed_Bandit_Problem.git
```

2. **Change into the project directory**

```bash
cd K_Armed_Bandit_Problem
```

3. **Run the simulation**

```bash
python main.py
```

4. Optionally modify parameters in `settings.py` (e.g., number of arms, steps, epsilon value) to experiment with behavior.

---

## 📈 Key Concepts

### Exploration vs. Exploitation

- **Exploration** means trying different actions to discover their reward potential.
- **Exploitation** means selecting the best-known action to maximize reward now.

Balancing the two is central to reinforcement learning strategies like ε-greedy. :contentReference[oaicite:1]{index=1}

---

## 🧪 Possible Extensions

Here are ideas to expand this project:

- Add additional strategies (e.g., Upper Confidence Bound, Thompson Sampling)
- Compare performance across multiple runs
- Add support for non-stationary reward distributions
- Add command-line arguments for settings

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and redistribute!

---

⭐ If you find this useful, please consider adding a ⭐ to the repository!  


The agent module contains the class Agent and two subclasses GreedyAgent and EpsilonGreedyAgent.

The graph module separates the codes of graphing from the main program.
