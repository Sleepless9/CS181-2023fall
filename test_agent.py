import gym
import os
import time
import numpy as np
from DDQN import DDQN_Agent
from DDQN import convert_greyscale
from DDQN import MAX_PENALTY, RENDER, REWARD_DIR, PRETRAINED_PATH

CONSECUTIVE_NEG_REWARD = 300

def test_agent(agent:DDQN_Agent, env:gym.make, model:str, test_num:int):
    agent.load(model)   # load the model
    all_rewards = []

    for test_index in range(test_num):
        state = env.reset() # get the state with color
        state_, _, _ = convert_greyscale(state) # change the state into colorless

        done = False        # whether the game is terminated
        reward = 0.00       # the total reward of a single game
        t1 = time.time()    # timer
        consecutive_negtive_reward = 0   # initialize 

        while not done and reward > MAX_PENALTY:
            action = agent.choose_action(state_, best=True) # choose the best action
            new_state, r, done, _ = env.step(action)        # get the next state according to action

            if RENDER:
                env.render()

            # if get negative reward for several consecutive frames
            consecutive_negtive_reward = update_neg_reward(consecutive_negtive_reward, r)
            if consecutive_negtive_reward >= 300:
                break
            
            new_state_, _, _ = convert_greyscale(new_state) # change next state into colorless
            state_ = new_state_     # update new state
            reward += r

        t1 = time.time() - t1
        all_rewards.append([reward, np.nan, t1, np.nan, np.nan])
        print(f"[INFO]: Run {test_index} | Run Reward: ", reward, " | Time: ", "%0.2fs."%t1)

    result_reward = []
    result_time = []
    for i in all_rewards:
        result_reward.append(i[0])
        result_time.append(i[2])

    reward_max, reward_min, reward_avg, reward_std = max(result_reward), min(result_reward), np.mean(result_reward), np.std(result_reward)
    time_avg = np.mean(result_time)

    all_rewards.append([reward_avg, np.nan, time_avg, reward_max, reward_min, reward_std])
    print(f"[INFO]: Run {test_index} | Avg Run Reward: ","%0.2f"%reward_avg, " | Avg Time: ",
          "%0.2fs"%time_avg, f" | Max: {reward_max} | Min: {reward_min} | Std Dev: {reward_std}")
    
    save_test_results(all_rewards)

    return [reward_avg, np.nan, time_avg, reward_max, reward_min, reward_std]


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
    np.savetxt(path, all_rewards, delimiter=",")


if __name__ == "__main__":
    env = gym.make('CarRacing-v0').env

    agent = DDQN_Agent()
    test_agent(agent, env, model=PRETRAINED_PATH, test_num=50)