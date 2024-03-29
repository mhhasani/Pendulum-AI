import os
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio


# path
ROOT_DIR = os.path.dirname(__file__)
Q_TABLE_PATH = os.path.realpath(os.path.join(ROOT_DIR, 'qtable.npy'))
GIF_PATH = os.path.realpath(os.path.join(ROOT_DIR, 'test.gif'))
STATISTIC_PATH = os.path.realpath(os.path.join(ROOT_DIR, 'Statistics.png'))

# constants
LEARNING_RATE = 0.4
DISCOUNT = 0.95
EPISODES = 20000
epsilon = 1
ACTION_SPACE_SIZE = 17
OBSERVATION_SPACE_SIZE = [21, 21, 65]

env = gym.make("Pendulum-v0")
ep_rewards = []
revards_log = {'ep': [], 'avg': [], 'max': [], 'min': []}
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / [i-1 for i in OBSERVATION_SPACE_SIZE]


def get_input():
    global EPISODES, epsilon
    learn = input("LEARN? (y/n): ")
    if learn == 'y':
        episode_count = input("Enter number of episodes: ")
        EPISODES = int(episode_count)
        return 'learn'
    else:
        epsilon = 0
        return 'test'


def make_action_space():
    high = env.action_space.high
    low = env.action_space.low
    make_action_space = (high - low) / (ACTION_SPACE_SIZE - 1)
    action_space = {}
    for i in range(ACTION_SPACE_SIZE):
        action_space[i] = [low[0] + (i * make_action_space[0])]
    return action_space


def create_qtable():
    if not os.path.exists(Q_TABLE_PATH):
        q_table = np.zeros(OBSERVATION_SPACE_SIZE + [ACTION_SPACE_SIZE])
        np.save(Q_TABLE_PATH, q_table)
    else:
        q_table = np.load(Q_TABLE_PATH)
    return q_table


def get_discrete_state(state):
    low = env.observation_space.low
    ds = (state - low) / discrete_os_win_size
    return tuple(ds.astype(np.int32))


def get_action(q_table, state, epsilon):
    if np.random.random() > epsilon:
        action = np.argmax(q_table[state])
    else:
        action = np.random.randint(0, ACTION_SPACE_SIZE)
    return action


def update_qtable(q_table, state, action, new_state, reward):
    current_q = q_table[state + (action, )]
    max_future_q = np.max(q_table[new_state])
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
        (reward + DISCOUNT * max_future_q)
    q_table[state + (action, )] = new_q
    return q_table


def q_learning():
    global epsilon
    q_table = create_qtable()
    action_space = make_action_space()

    for episode in range(EPISODES):
        episode_reward = 0
        state = get_discrete_state(env.reset())
        done = False
        while not done:
            action = get_action(q_table, state, epsilon)
            torque = action_space[action]
            new_state, reward, done, _ = env.step(torque)
            episode_reward += reward
            new_state = get_discrete_state(new_state)
            q_table = update_qtable(
                q_table, state, action, new_state, reward)
            state = new_state
        epsilon *= 0.999
        log(episode, episode_reward)

    np.save(Q_TABLE_PATH, q_table)
    return q_table


def log(episode, episode_reward, test=False):
    ep_rewards.append(episode_reward)
    if not episode % 100 or test:
        average_reward = sum(ep_rewards[-100:]) / 100
        revards_log['ep'].append(episode)
        revards_log['avg'].append(average_reward)
        revards_log['max'].append(max(ep_rewards[-100:]))
        revards_log['min'].append(min(ep_rewards[-100:]))
        print(
            f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.3f}")


def test_and_make_gif():
    q_table = create_qtable()
    action_space = make_action_space()
    chart_data_time = []
    chart_data_reward = []
    chart_data_action = []
    images = []
    for episode in range(1):
        episode_reward = 0
        state = get_discrete_state(env.reset())
        done = False
        while not done:
            action = get_action(q_table, state, 0)
            torque = action_space[action]
            new_state, reward, done, _ = env.step(torque)
            episode_reward += reward
            new_state = get_discrete_state(new_state)
            state = new_state
            images.append(env.render(mode='rgb_array'))
            chart_data_time.append(action)
            chart_data_reward.append(reward)
            chart_data_action.append(torque)
        log(episode, episode_reward, True)

    env.close()
    print("Making gif...")
    imageio.mimsave(GIF_PATH, images)
    draw_3D_chart(chart_data_time, chart_data_reward, chart_data_action)


def show_statistics():
    ep = revards_log['ep']
    avg = revards_log['avg']
    max = revards_log['max']
    min = revards_log['min']
    with open("statistics", 'wb') as filehandler:
        pickle.dump(revards_log, filehandler)
        plt.plot(ep, avg, label="average rewards")
        plt.plot(ep, max, label="max rewards")
        plt.plot(ep, min, label="min rewards")
        plt.legend(loc=4)
        plt.savefig(STATISTIC_PATH)
        plt.show()


def draw_3D_chart(chart_data_time, chart_data_reward, chart_data_action):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(chart_data_time, chart_data_reward,
               chart_data_action, c='r', marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.set_zlabel('Action')
    plt.show()
    # save it in a image file
    ax.figure.savefig('QLearning_chart.png')


def main():
    learn_or_test = get_input()
    if learn_or_test == 'learn':
        q_learning()
        show_statistics()
    else:
        test_and_make_gif()


if __name__ == "__main__":
    main()
