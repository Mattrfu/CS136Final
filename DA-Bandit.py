"""A very simple contextual bandit example with 3 arms."""

import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import random
import copy

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

class Person:
    # constructor to initialize the attributes of Person class
    def __init__(self, name, preferences):
        self.name = name
        self.partner = None
        self.preferences = preferences

    # return object representation
    def __repr__(self):
        if self.partner:
            return f'{self.name} ⚭ {self.partner}'
        else:
            return f'{self.name} ⌀'


class Alpha(Person):
    def __init__(self, name, preferences):
        # super() refers to parent class, and inherits methods
        super().__init__(name, preferences)
        # prefered person not asked yet
        # recursively copy
        self.not_asked = copy.deepcopy(preferences)

    def ask(self):
        # drop the first element which is the next preferred person
        return self.not_asked.pop(0)

    # for check_stability function
    def accept(self, suitor):
        return self.partner is None or(
            # check that the suitor is strictly preferred to the existing partner
            self.preferences.index(suitor) <
            self.preferences.index(self.partner)
        )


class Beta(Person):
    def __init__(self, name, preferences):
        super().__init__(name, preferences)
        # this person does not ask

    def accept(self, suitor):
        return self.partner is None or(
            # check that the suitor is strictly preferred to the existing partner
            self.preferences.index(suitor) <
            self.preferences.index(self.partner)
        )

def setupDA(context, desires, agents):
    global alphas
    global betas

    alphas = [None] * agents
    alphas[0] = Alpha(0, context)
    for i in range(1, agents):
        alphas[i] = Alpha(i, desires[i])

    betas = [None] * agents
    for j in range(agents, agents * 2):
        betas[j] = Beta(j, desires[j])

def runDA(context, desires, agents):
    setupDA(context, desires, agents)
    print("Proposing: ", alphas)
    print("Accepting: ", betas)
    print()

    # all alphas are unmatched at the beginning
    unmatched = list(alphas.keys())

    while unmatched:
        # randomly select one of the alphas to choose next
        alpha = alphas[random.choice(unmatched)]
        # alpha ask his first choice
        temp = alpha.ask()
        if temp == None:
            break
        beta = betas[temp]
        print(f'{alpha.name} asks {beta.name}')
        # if beta accepts alpha's proposal
        if beta.accept(alpha.name):
            print(f'{beta.name} accepts')
            # # if beta has a partner
            if beta.partner:
                # this existing alpha partner is now an ex
                ex = alphas[beta.partner]
                print(f'{beta.name} dumps {ex.name}')
                # this alpha person has no partner now :(
                ex.partner = None
                # add this alpha person back to the list of unmatched
                unmatched.append(ex.name)
            unmatched.remove(alpha.name)
            # log the match
            alpha.partner = beta.name
            beta.partner = alpha.name
        else:
            print(f'{beta.name} rejects')
            # move on to the next unmatched male
    print()
    print("Everyone is matched. This is a stable matching")
    print(alphas)
    print(betas)

    return alphas[0].partner



class DABandit(gym.Env):
    def __init__(self, config=None):
        self.desires = []
        for i in range(config["num-agents"] * 2):
            self.desires += [random.shuffle(list(range(1, config["num-agents"])))]
        self.action_space = Discrete(config["num-agents"] + 1)
        self.observation_space = self.desires[0]
        self.cur_context = None

    def reset(self):
        self.cur_context = []
        self.desires = []
        for i in range(config["num-agents"] * 2):
            self.desires += [random.shuffle(list(range(1, config["num-agents"])))]
        self.action_space = Discrete(config["num-agents"] + 1)
        self.observation_space = self.desires[0]
        return np.array(self.observation_space)

    def step(self, action):
        reward = 0
        if action == len(self.action_space) - 1:
            # This is truncation case
            if len(set(self.cur_context)) > len(self.cur_context):
                reward = -10
            else:
                partner = runDA(self.cur_context, self.desires, config["num-agents"])
                if partner == None:
                    reward = -1
                else:
                    reward = len(self.desires[0]) - self.desires[0].index(partner)


        return (np.array(self.cur_context), reward, True,
                {
                    "regret": 10 - reward
                })


if __name__ == "__main__":
    ray.init(num_cpus=3)
    args = parser.parse_args()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    config = {
        "env": DABandit,
        "num-agents": args.num-agents,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)