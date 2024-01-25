import gym
import os
import time
import numpy as np
from DDQN_self import Agent
from DDQN_self import convert_to_grey

MAX_PENALTY = -5
RENDER = True
CONSECUTIVE_NEG_REWARD = 300
REWARD_DIR = "./reword_self"
PRETRAINED_PATH = "./model_self/episode_500.h5"

def test(agent:Agent, env:gym.make, model:str, test_num:int):
    agent.load_model(model)   # load the model
    all_rewards = []
    result_reward = []
    result_time = []

    for test_index in range(test_num):
        state = env.reset() # get the state with color
        state_, _ = convert_to_grey(state) # change the state into colorless

        done = False        # whether the game is terminated
        reward = 0.00       # the total reward of a single game
        t1 = time.time()    # timer
        consecutive_negtive_reward = 0   # initialize 

        while True:
            action = agent.choose_action(state_, True) # choose the best action
            new_state, r, done, _ = env.step(action)        # get the next state according to action

            if RENDER:
                env.render()

            # if get negative reward for several consecutive frames
            consecutive_negtive_reward = update_neg_reward(consecutive_negtive_reward, r)
            if consecutive_negtive_reward >= CONSECUTIVE_NEG_REWARD:
                break
            
            new_state_, _ = convert_to_grey(new_state) # change next state into colorless
            state_ = new_state_     # update new state
            reward += r

            if done or reward <= MAX_PENALTY:
                break 

        t1 = time.time() - t1
        reward = round(reward, 2)
        t1 = round(t1, 2)

        all_rewards.append([reward, t1])

        result_reward.append(reward)
        result_time.append(t1)
        print(f"[INFO]: Run {test_index} | Run Reward: {reward} | Time: {t1}s.")

    reward_max, reward_min, reward_avg, reward_std = \
        max(result_reward), min(result_reward), round(np.mean(result_reward), 2), round(np.std(result_reward), 2)
    time_avg = round(np.mean(result_time), 2)

    all_rewards.append([reward_avg, time_avg, reward_max, reward_min, reward_std])

    print(f"[INFO]: Total Run {test_index+1} | Avg Run Reward: {reward_avg} | Avg Time: {time_avg} | Max: {reward_max} | Min: {reward_min} | Std Dev: {reward_std}")
    
    save_test_results(all_rewards)

    # return [reward_avg, time_avg, reward_max, reward_min, reward_std]


def update_neg_reward(consecutive_negtive_reward, r):
    if r < 0:
        return consecutive_negtive_reward + 1
    else:
        return 0

def save_test_results(all_rewards):
    save_path = f"test_{REWARD_DIR}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = save_path + PRETRAINED_PATH.split('/')[-1][:-3] + "_run_rewards.csv"
    np.savetxt(path, all_rewards, delimiter=",", fmt = "%s")


if __name__ == "__main__":
    env = gym.make('CarRacing-v0').env

    agent = Agent()
    test(agent, env, model=PRETRAINED_PATH, test_num=20)