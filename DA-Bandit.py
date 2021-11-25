"""A very simple contextual bandit example with 3 arms."""

import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import random

import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="contrib/LinUCB")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--stop-reward", type=float, default=10.0)


class DABandit(gym.Env):
    def __init__(self, config=None):
        desires = []
        for i in range(config["num-agents"] * 2):
            desires += [random.shuffle(list(range(1, config["num-agents"])))]
        self.action_space = Discrete(config["num-agents"] + 1)
        self.observation_space = desires[0]
        self.cur_context = None

    def reset(self):
        self.cur_context = []
        return np.array(self.cur_context)

    def step(self, action):
        rewards_for_context = {
            -1.: [-10, 0, 10],
            1.: [10, 0, -10],
        }
        reward = rewards_for_context[self.cur_context][action]
        return (np.array([-self.cur_context, self.cur_context]), reward, True,
                {
                    "regret": 10 - reward
                })


if __name__ == "__main__":
    ray.init(num_cpus=3)
    args = parser.parse_args()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    config = {
        "env": DABandit,
        "num-agents": args.num-agents,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)