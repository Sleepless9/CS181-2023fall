import gym
from tqdm import tqdm
from DDQN_self import Agent
from DDQN_self import convert_to_grey

MAX_PENALTY = -5
RENDER = True
CONSECUTIVE_NEG_REWARD = 100
SKIP_FRAMES = 2
UPDATE_STEPS = 5
SAVE_FREQUENCY = 100

def train(agent:Agent, env:gym.make, episode_num:int):
    all_rewards = []

    for episode in tqdm(range(episode_num)):
        print(f"[INFO]: Episode {episode}")
        state = env.reset() # get the state with color
        state_, road_visibility = convert_to_grey(state) # change the state into colorless

        done = False        # whether the game is terminated
        sum_reward = 0.00       # the total reward of a single game
        consecutive_negtive_reward = 0   # initialize 

        while not done and sum_reward > MAX_PENALTY and road_visibility:
            action, new_state, reward = choose_and_take_action(agent, env, state_)

            # if get negative reward for several consecutive frames
            consecutive_negtive_reward = update_neg_reward(consecutive_negtive_reward, reward)
            if consecutive_negtive_reward >= CONSECUTIVE_NEG_REWARD:
                break

            new_state_, road_visibility = convert_to_grey(new_state)

            # limit the reward because we don't want car run to fast
            if reward > 1:
                reward = 1
            if reward < -10:
                reward = -10

            agent.store_transition(state_, action, reward, new_state_, done)
            agent.experience()

            # update
            state_ = new_state_
            sum_reward += reward
        
        all_rewards.append([sum_reward, agent.epsilon])

        update_target_and_save(agent, episode, all_rewards)

    env.close()


def choose_and_take_action(agent, env, state_):
    reward = 0
    action = agent.choose_action(state_) # choose the best action
    for _ in range(SKIP_FRAMES + 1):
        new_state, r, done, _ = env.step(action)        # get the next state according to action
        reward += r

        if RENDER:
            env.render()

        if done:
            break
    
    return action, new_state, reward

def update_neg_reward(consecutive_negtive_reward, r):
    if r < 0:
        return consecutive_negtive_reward + 1
    else:
        return 0

def update_target_and_save(agent, episode, all_rewards):
    if episode % UPDATE_STEPS == 0:
        agent.update_model()

    if episode % SAVE_FREQUENCY == 0:
        agent.save_model(episode, data=all_rewards)


if __name__ == "__main__":
    env = gym.make('CarRacing-v0').env

    agent = Agent()
    train(agent, env, episode_num=1500)
