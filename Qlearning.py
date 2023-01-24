import os
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio


# constants
ROOT_DIR = os.path.dirname(__file__)
IS_LEARNING = True
LEARNING_RATE = 0.4
DISCOUNT = 0.95
EPISODES = 20000
epsilon = 1
ACTION_SPACE_SIZE = 17
OBSERVATION_SPACE_SIZE = [21, 21, 65]
env = gym.make("Pendulum-v0")
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / [i-1 for i in OBSERVATION_SPACE_SIZE]


def get_input():
    global IS_LEARNING, EPISODES, epsilon
    learn = input("LEARN? (y/n): ")
    if learn == 'y':
        IS_LEARNING = True
        episode_count = input("Enter number of episodes: ")
        EPISODES = int(episode_count)
    else:
        IS_LEARNING = False
        epsilon = 0


def make_action_space():
    make_action_space = (
        env.action_space.high - env.action_space.low) / (ACTION_SPACE_SIZE - 1)
    action_space = {}
    for i in range(ACTION_SPACE_SIZE):
        action_space[i] = [env.action_space.low[0] +
                           (i * make_action_space[0])]
    return action_space


def create_qtable():
    if not os.path.exists(os.path.realpath(os.path.join(ROOT_DIR, 'qtable.npy'))):
        q_table = np.zeros(OBSERVATION_SPACE_SIZE + [ACTION_SPACE_SIZE])
        np.save(os.path.realpath(os.path.join(ROOT_DIR, 'qtable.npy')), q_table)
    else:
        q_table = np.load(os.path.realpath(
            os.path.join(ROOT_DIR, 'qtable.npy')))
    return q_table


def get_discrete_state(state):
    ds = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(ds.astype(np.int32))


def get_action(q_table, discrete_state, epsilon):
    if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = np.random.randint(0, ACTION_SPACE_SIZE)
    return action


def update_qtable(q_table, discrete_state, action, new_discrete_state, reward):
    current_q = q_table[discrete_state + (action, )]
    max_future_q = np.max(q_table[new_discrete_state])
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
        (reward + DISCOUNT * max_future_q)
    q_table[discrete_state + (action, )] = new_q
    return q_table


def save_qtable(q_table):
    np.save(os.path.realpath(os.path.join(ROOT_DIR, 'qtable.npy')), q_table)


def q_learning():
    global epsilon
    q_table = create_qtable()
    action_space = make_action_space()

    for episode in range(EPISODES):
        episode_reward = 0
        discrete_state = get_discrete_state(env.reset())
        done = False
        while not done:
            action = get_action(q_table, discrete_state, epsilon)
            torque = action_space[action]
            new_state, reward, done, _ = env.step(torque)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)
            q_table = update_qtable(
                q_table, discrete_state, action, new_discrete_state, reward)
            discrete_state = new_discrete_state
        epsilon *= 0.999
        log(episode, episode_reward)

    save_qtable(q_table)
    return q_table


def log(episode, episode_reward, test=False):
    ep_rewards.append(episode_reward)
    if not episode % 100 or test:
        average_reward = sum(
            ep_rewards[-100:]) / 100
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-100:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-100:]))
        print(
            f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.3f}")


def test_and_make_gif():
    q_table = create_qtable()
    action_space = make_action_space()
    images = []
    for i in range(5):
        episode_reward = 0
        discrete_state = get_discrete_state(env.reset())
        done = False
        while not done:
            action = np.argmax(q_table[discrete_state])
            torque = action_space[action]
            new_state, reward, done, _ = env.step(torque)
            episode_reward += reward
            discrete_state = get_discrete_state(new_state)
            images.append(env.render(mode='rgb_array'))
        log(i, episode_reward, test=True)

    env.close()
    # 30 fps
    print("Saving gif...")
    imageio.mimsave(os.path.realpath(
        os.path.join(ROOT_DIR, 'test.gif')), images, fps=30)


def show_statistics():
    with open("statistics", 'wb') as filehandler:
        pickle.dump(aggr_ep_rewards, filehandler)
        plt.plot(aggr_ep_rewards['ep'],
                 aggr_ep_rewards['avg'], label="average rewards")
        plt.plot(aggr_ep_rewards['ep'],
                 aggr_ep_rewards['max'], label="max rewards")
        plt.plot(aggr_ep_rewards['ep'],
                 aggr_ep_rewards['min'], label="min rewards")
        plt.legend(loc=4)
        plt.savefig(os.path.realpath(os.path.join(ROOT_DIR, "Statistics.png")))
        plt.show()


def main():
    get_input()
    if IS_LEARNING:
        q_learning()
        show_statistics()
    else:
        test_and_make_gif()


if __name__ == "__main__":
    main()
