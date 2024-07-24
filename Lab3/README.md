# LABORATORY 3 - DRL

#Video Results: 

## Cartpole REINFORCE without BASELINE

https://github.com/user-attachments/assets/db866774-17cb-4ce2-81ae-c309279ea308.mp4

## Cartpole REINFORCE with BASELINE

https://github.com/user-attachments/assets/ecd8d3b5-ebb4-4f57-a44f-a10ddc36c93f.mp4

## Cartpole DQN

https://github.com/user-attachments/assets/fed700e4-c9f1-4d59-bfb7-b36ac5f87c89.mp4

## LunarLender DQN

https://github.com/user-attachments/assets/7c3eae00-479d-4d07-b11e-375640a6aa71.mp4

## CarRacing DQN
Example of the CarRacing DQN algorithm working.

![CarRacing DQN](images/example_DQNCarRacing.gif)

## Folder structure

- `Lab3-DRL.ipynb`: Base Jupyter notebook with the laboratory instructions from the course.
- `Lab3-DRL-PARTLY.ipynb`: Jupyter notebook with the exercises completed by the student during the laboratory session.
- `Lab3-DRL-Exercise3.ipynb`: Jupyter notebook with the exercises completed by the student for the exam.
- `wandb`: Folder with the weights and biases files generated during the training of the models.
- `images`: Folder with the images used in the README file.
- `model_states`: Folder with the model states generated during the training of the models.
  
# 1. Introduction

This laboratory is focused on Deep Reinforcement Learning (DRL). The main goal is to understand the basic concepts of DRL and how to implement them using the PyTorch library. 

# 2. Exercises

We have three different exercises in this laboratory:

1. For the first exercise it is necessary to implement the REINFORCE algorithm to solve the LunarLander-v2 environment from the OpenAI Gym.

2. For the second exercise it is necessary to implement the DQN algorithm to solve the CartPole-v1 and LunarLander-v2 environments from the OpenAI Gym.

3. For the third exercise it is necessary to implement the PPO algorithm to solve the CarRacing-v0 environment from the OpenAI Gym.

## 2.1. Exercise 1

For the first exercise, I implemented the REINFORCE algorithm to solve the LunarLander-v2 and CartPole-v1 environments from the OpenAI Gym with and without a baseline.

### REINFORCE Algorithm

The REINFORCE algorithm is a policy gradient method that estimates the policy gradient using the likelihood ratio trick. The policy gradient is defined as the gradient of the expected return with respect to the policy parameters.

![Results: REINFORCE](images/REINFORCE-formula.png)

### Results

#### CartPole-v1 [without baseline]
![Results: REINFORCE for Cartpole](images/cartpole_no_baseline.png)

#### LunarLander-v2 [without baseline]
![Results: REINFORCE for LunarLander](images/lunalender_no_baseline.png)

#### CartPole-v1 [with baseline]
![Results: REINFORCE for Cartpole](images/cartpole_baseline.png)

#### LunarLander-v2 [with baseline]
![Results: REINFORCE for LunarLander](images/lunalender_baseline.png)

We can see that the REINFORCE algorithm with a baseline has a better performance than the REINFORCE algorithm without a baseline. 

The REINFORCE algorithm, with and without a baseline, is not able to solve the LunarLander-v2 environment, reaching a maximum score of around 170 points for the REINFORCE algorithm with a baseline.

## 2.2. Exercise 2

For the second exercise, I implemented the DQN algorithm to solve the CartPole-v1 and LunarLander-v2 environments from the OpenAI Gym.

### DQN Algorithm

The DQN algorithm is a model-free, off-policy algorithm for learning action-value functions. The DQN algorithm uses a replay buffer to store the experiences and a target network to stabilize the learning process.

![Results: DQN](images/DQN-algorithm.png)

### Results

#### CartPole-v1
![Results: DQN for CartPole-v1](images/dqn_cartpole.png)

#### LunarLander-v2
![Results: DQN for LunarLander-v2](images/dqn_lunalender.png)

In this case, the DQN algorithm was able to solve the LunarLander-v2 environments. The DQN algorithm was able to achieve a good performance in both environments after few episodes of training respect to the REINFORCE algorithm.

## 2.3. Exercise 3 

For the third exercise, I implemented the PPO algorithm to solve the CarRacing-v0 environment from the OpenAI Gym.
In this case, the network architecture is a CNN with two convolutional layers and two fully connected layers.

The main problem for this exercise was the training time. 

For the environment CarRacing-v2, I implemented a wrapper to preprocess the images, resizing the images, converting them to grayscale, and skipping frames.

### DQN Algorithm
I try to resolve the CarRacing-v2 environment with the DQN algorithm.

#### Results
In the notebook cell, where the DQN algorithm is implemented, seems to be that the algorithm is working correctly.

Eval Reward         |  Total Reward
:-------------------------:|:-------------------------:
![DQN CarRacing eval reward](images/DQN_carracing_total_reward.png) | ![DQN CarRacing total reward](images/DQN_carracing_tot_reward.png)

![DQN CarRacing episode](images/DQN_carracing%20episodes.png)

### PPO Algorithm
The PPO algorithm is a policy gradient method that uses a clipped objective function to improve the stability of the learning process. 

![PPO Algorithm](images/PPO.png)

#### Results - Base and Improved Implementation
Like the DQN algorithm, the implementation of the PPO algorithm seems to be working correctly.


