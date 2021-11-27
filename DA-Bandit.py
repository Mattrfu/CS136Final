"""A very simple contextual bandit example with 3 arms."""

import argparse
import gym
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import numpy as np
import random
import copy

from ray.rllib.agents import ppo

import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="contrib/LinUCB")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=20000)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--stop-reward", type=float, default=50.0)

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

def runStudentDA(cur_context, input_desires, num_agents):
    desires = copy.deepcopy(input_desires)
    desires[0] = cur_context

    print("Proposing preferences: ", desires[:num_agents])
    print("Accepting preferences: ", desires[num_agents:])
    print()

    student_preferences = copy.deepcopy(desires[:num_agents])
    teacher_preferences = desires[num_agents:]
    
    student_partners = [None] * num_agents
    teacher_partners = [None] * num_agents
    
    counter = 1
    while None in student_partners:

        # Step 1: All student ask their top preference remaining beta
        for student in range(0, num_agents):
            if student_partners[student] == None: # student is unassigned a teacher
                top_preference = student_preferences[student].pop(0)

                if top_preference == None:
                    # Student prefers themselves
                    student_partners[student] == -1
                    break

                # Step 2: All teachers accept their top preferred suitor, rejecting existing suitor
                # print("Student " + str(student)  + " proposes to Teacher " + str(top_preference))

                # if beta accepts alpha's proposal
                if teacher_partners[top_preference] is None or (
                    # check that the suitor is strictly preferred to the existing partner
                    teacher_preferences[top_preference].index(student) <
                    teacher_preferences[top_preference].index(teacher_partners[top_preference])
                ):

                    # if beta has a partner
                    if teacher_partners[top_preference]:
                        # this existing alpha partner is now an ex
                        # print('Teacher ' + str(top_preference) + ' rejects  Student ' + str(teacher_partners[top_preference]))
                        # this alpha person has no partner now :(
                        student_partners[teacher_partners[top_preference]] = None
            
                    # log the match
                    # print('Teacher ' + str(top_preference) + ' accepts  Student ' + str(student))
                    student_partners[student] = top_preference
                    teacher_partners[top_preference] = student
                # else:
                    # print('Teacher ' + str(top_preference) + ' rejects  Student ' + str(student))
                    # move on to the next unmatched student
        
        print("Teacher-Proposing DA Round " + str(counter) + " results -- (student, teacher)")
        counter += 1
        for i in range(0, len(student_partners)):
            print("(" + str(i) + "," + str(student_partners[i]) + ")")

    print()
    print("Everyone is matched. This is a stable matching")

    # return score of Student 1's matching
    student_1_matched_teacher = student_partners[0]
    if student_1_matched_teacher == -1: # matched themselves
        return 0

    return num_agents - desires[:num_agents][0].index(student_1_matched_teacher)

def runTeacherDA(cur_context, input_desires, num_agents):
    desires = copy.deepcopy(input_desires)
    desires[0] = cur_context

    print("Proposing preferences: ", desires[:num_agents])
    print("Accepting preferences: ", desires[num_agents:])
    print()

    student_preferences = copy.deepcopy(desires[:num_agents])
    teacher_preferences = desires[num_agents:]
    
    student_partners = [None] * num_agents
    teacher_partners = [None] * num_agents
    
    counter = 1
    while None in teacher_partners:

        # Step 1: All teachers ask their top preference remaining student
        for teacher in range(0, num_agents):
            if teacher_partners[teacher] == None: # teacher is unassigned a partner
                top_preference = teacher_preferences[teacher].pop(0)

                if top_preference == None:
                    # Teacher prefers themselves
                    teacher_partners[teacher] == -1
                    break

                # Step 2: All beta's accept their top preferred suitor, rejecting existing suitor
                # print("Student " + str(student)  + " proposes to Teacher " + str(top_preference))

                # if beta accepts alpha's proposal
                if student_partners[top_preference] is None or (
                    # check that the suitor is strictly preferred to the existing partner
                    student_preferences[top_preference].index(teacher) <
                    student_preferences[top_preference].index(student_partners[top_preference])
                ):

                    # # if beta has a partner
                    if student_partners[top_preference]:
                        # this existing alpha partner is now an ex
                        # print('Teacher ' + str(top_preference) + ' rejects  Student ' + str(teacher_partners[top_preference]))
                        # this alpha person has no partner now :(
                        teacher_partners[student_partners[top_preference]] = None
            
                    # log the match
                    # print('Teacher ' + str(top_preference) + ' accepts  Student ' + str(student))
                    teacher_partners[teacher] = top_preference
                    student_partners[top_preference] = teacher
                # else:
                    # print('Teacher ' + str(top_preference) + ' rejects  Student ' + str(student))
                    # move on to the next unmatched teacher
        
        print("Teacher-Proposing DA Round " + str(counter) + " results -- (student, teacher)")
        counter += 1
        for i in range(0, len(teacher_partners)):
            print("(" + str(i) + "," + str(student_partners[i]) + ")")

    print()
    print("Everyone is matched. This is a stable matching")

    # return score of Student 1's matching
    student_1_matched_teacher = student_partners[0]
    if student_1_matched_teacher == -1: # matched themselves
        return 0

    return num_agents - desires[:num_agents][0].index(student_1_matched_teacher)

class DABandit(gym.Env):
    def __init__(self, config=None):
        self.num_agents = config["num-agents"]
        self.desires = []
        for i in range(config["num-agents"] * 2):
            tempList = list(range(0, config["num-agents"]))
            random.shuffle(tempList)
            self.desires += [tempList]
        print(self.desires)
        self.action_space = Discrete(config["num-agents"] + 1)
        self.observation_space = MultiDiscrete([config["num-agents"] + 1] * config["num-agents"] * 2)
            #MultiDiscrete(tuple([len(self.desires[0])] * len(self.desires[0]) * 2))
        #     Box(
        #     self.desires[0], shape=(config["num-agents"],), dtype=np.float32
        # )

            #Tuple(self.desires[0])
        self.cur_context = None

    def reset(self):
        #print("Happens")
        self.cur_context = []
        self.desires = []
        for i in range(self.num_agents * 2):
            tempList = list(range(0, self.num_agents))
            random.shuffle(tempList)
            self.desires += [tempList]
        return np.array(self.desires[0] + self.cur_context + [self.num_agents] * (self.num_agents - len(self.cur_context)))

    def step(self, action):
        Done = False
        # print("Step occuring")
        # print(self.desires[0])
        # print(action)

        # print(self.cur_context)
        reward = 0
        if action in self.cur_context:
            Done = True
            reward = -100
            #self.reset()
            self.cur_context = []
        elif action == self.num_agents:
            Done = True
            # This is truncaction case

            # partner = runDA(self.cur_context, self.desires, self.num_agents)
            # if partner == None:
            #     reward = -1
            # else:
            #     reward = len(self.desires[0]) - self.desires[0].index(partner)
            #print(self.cur_context)
            #print(self.desires[0])
            # happens = False
            # for i in range(len(self.cur_context)):
            #     if self.cur_context[i] == self.desires[0][i]:
            #         reward += (len(self.desires) - i) * 100
            #         happens = True
            # if not happens:
            #     reward = -10

            # Run Deferred Acceptance Matching
            # Run Student-Proposing DA
            reward = runStudentDA(self.cur_context, self.desires, self.num_agents)
            
            # Run Teacher-Proposing DA
            # reward = runTeacherDA(self.cur_context, self.desires, self.num_agents)
            
            
            f = open("results1.txt", "a")
            #f.write("Context: " + ",".join(str(e) for e in self.cur_context) + ". Desires: " + ", ".join(str(e) for e in self.desires[0]) + "\n")
            f.write(str(self.cur_context == self.desires[0]) + "\n")
            f.close()
            # self.cur_context = []
            # print("happens1")
            #self.reset()
            self.cur_context = []
        else:
            self.cur_context += [action]

        return (np.array(self.desires[0] + self.cur_context + [self.num_agents] * (self.num_agents - len(self.cur_context))), reward, Done,
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

        # "rollout_fragment_length": 1,
        # "train_batch_size": 1,
        # "timesteps_per_iteration": 100,
        "env_config": {"num-agents": args.num_agents},
    }

    # trainer = ppo.PPOTrainer()
    # while True:
    #     print(trainer.train)
    results = tune.run(ppo.PPOTrainer, config=config, stop=stop)

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)