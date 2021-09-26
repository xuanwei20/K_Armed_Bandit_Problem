# Michael added a readme md!

using:
```sh
$ nano Readme.md
```

Description of the Code
Multi-armed bandit problem's main module starts a class called ExplorationVsExploitation. Settings is imported into the main program file; MultiArmedBandit is imported to make an instance of MultiArmedBandit; GreedyAgent, EpsilonGreedyAgent are imported to make an instance of GreedyAgent and an instance of EpsilonGreedyAgent; draw_graph function is imported from graph module.
The settings module contains a class called Settings to store all the values in one place instead of adding settings throughout the code. Simply changing some values in settings.py can modify the multi-armed bandit problem.
The multi_armed_bandit module contains the class MultiArmedBandit.
The agent module contains the class Agent and two subclasses GreedyAgent and EpsilonGreedyAgent.
The graph module separates the codes of graphing from the main program.
