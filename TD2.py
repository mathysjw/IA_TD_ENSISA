import numpy as np
import random

class TaxiEnv:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.locs = {
            0: (0, 0),
            1: (0, 4),
            2: (4, 0),
            3: (4, 3)   
        }

        self.n_actions = 6
        self.n_states  = 5 * 5 * 5 * 4 

    def encode(self, tr, tc, pass_loc, dest):
        i = tr
        i = i * 5 + tc
        i = i * 5 + pass_loc
        i = i * 4 + dest
        return i

    def decode(self, s):
        dest = s % 4; s //= 4
        pass_loc = s % 5; s //= 5
        tc = s % 5; s //= 5
        tr = s
        return tr, tc, pass_loc, dest

    def reset(self):
        tr = random.randint(0,4)
        tc = random.randint(0,4)
        pass_loc = random.randint(0,3)
        dest = random.randint(0,3)
        self.state = (tr, tc, pass_loc, dest)
        return self.encode(*self.state)

    def step(self, action):
        tr, tc, pass_loc, dest = self.state
        reward = -1
        done = False

        if action == 0 and tc > 0:
            tc -= 1
        elif action == 1 and tc < 4:
            tc += 1
        elif action == 2 and tr > 0:
            tr -= 1
        elif action == 3 and tr < 4:
            tr += 1

        elif action == 4:
            if pass_loc < 4 and (tr,tc) == self.locs[pass_loc]:
                pass_loc = 4
            else:
                reward = -10

        elif action == 5:
            if pass_loc == 4 and (tr,tc) == self.locs[dest]:
                reward = +20
                pass_loc = dest
                done = True
            else:
                reward = -10

        self.state = (tr, tc, pass_loc, dest)
        return self.encode(*self.state), reward, done, {}


def q_learning(env, episodes=20000, alpha=0.9, gamma=0.99,
               eps_start=1.0, eps_end=0.05, decay=15000):

    Q = np.zeros((env.n_states, env.n_actions))
    eps = eps_start
    eps_decay = (eps_start - eps_end) / decay

    for ep in range(episodes):
        s = env.reset()
        done = False

        while not done:
            if random.random() < eps:
                a = random.randint(0, env.n_actions-1)
            else:
                a = np.argmax(Q[s])

            s2, r, done, _ = env.step(a)

            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            s = s2

        eps = max(eps_end, eps - eps_decay)

    return Q

env = TaxiEnv()
Q = q_learning(env)

def evaluate(env, Q, episodes=1000):
    total = 0
    success = 0

    for _ in range(episodes):
        s = env.reset()
        done = False
        r_tot = 0

        while not done:
            a = np.argmax(Q[s])
            s, r, done, _ = env.step(a)
            r_tot += r

        total += r_tot
        if r == 20:
            success += 1

    return total / episodes, success / episodes

avg_reward, success_rate = evaluate(env, Q)

print("Average reward :", avg_reward)
print("Success rate   :", success_rate)
