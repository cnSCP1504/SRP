# import LCRL code trainer module
from test_test import train
# import the pre-built LDBA for minecraft-t1
from lcrl.src.automata.minecraft_1 import minecraft_1
from minecraft_test import minecraft_test
# import the pre-built MDP for minecraft-t1
from lcrl.src.environments.minecraft import minecraft

LDBA = minecraft_test
MDP = minecraft

successes = 0

while successes < 0.8:
     # train the agent
    task = train(MDP, LDBA,
                algorithm='ql',
                episode_num=1000,
                iteration_num_max=4000,
                discount_factor=0.9,
                learning_rate=0.9
                )
    successes = task.successes_in_test /100