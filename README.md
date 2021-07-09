# Count-Based Exploration for Multi-Agent Learning
This is the code for implementing Simple and Centralized Count-Based in the Master Thesis: 
The multi-agent environment is from https://github.com/openai/multiagent-particle-envs and we make some changes for our experiments. 
The DDPG code is from stable-basedline3 https://github.com/DLR-RM/stable-baselines3 and we extend it to Simple MADDPG and Centralized MADDPG. 
The detailed explaination of both the exploration strategies and the multi-agent learning algorithms are in the report.

## Installation
To install the environments, `cd` to the ma_envs directory, and `pip install -e .`. More information of the environments can see link.
And check if you fullfill the requirements of stable-baselines like python3 (>=3.5), PyTorch...

## Training
Train Centralized MADDPG agents with the help of Centralized Count-Based exploration on the Communicative Navigation task: 
```
python stable_baselines3/tests/test_maddpg_countbased.py
```
