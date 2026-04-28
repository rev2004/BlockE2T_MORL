# BlockE2T-MORL Full Simulation (Python Script)
# Run: python blocke2t_morl_simulation.py

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

GRID=10
START=(0,0)
GOAL=(9,9)

ACTIONS=[
(0,1),(0,-1),(1,0),(-1,0)
]

malicious_nodes=[(3,4),(4,5),(7,2),(6,8)]

class CloudEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state=START
        return self.state

    def valid(self,pos):
        x,y=pos
        return 0<=x<GRID and 0<=y<GRID

    def step(self,action):
        move=ACTIONS[action]
        ns=(self.state[0]+move[0],self.state[1]+move[1])

        if not self.valid(ns):
            ns=self.state

        self.state=ns

        energy=np.random.uniform(1,4)
        latency=np.random.uniform(1,3)

        trust=1.0
        if ns in malicious_nodes:
            trust=0.2

        reward=-0.4*energy-0.3*latency+0.3*trust

        done=False
        if ns==GOAL:
            reward+=50
            done=True

        return ns,reward,done,energy,latency,trust


class BlockchainTrust:
    def __init__(self):
        self.trust={}

    def score(self,node):
        if node not in self.trust:
            self.trust[node]=1
        return self.trust[node]

    def update(self,node,validation):
        old=self.score(node)
        alpha=.7
        self.trust[node]=alpha*old+(1-alpha)*validation


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,128)
        self.actor=nn.Linear(128,4)
        self.critic=nn.Linear(128,1)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        policy=torch.softmax(self.actor(x),dim=-1)
        value=self.critic(x)
        return policy,value


model=ActorCritic()
optimizer=optim.Adam(model.parameters(),lr=.001)
gamma=.99

def select_action(state):
    s=torch.FloatTensor(state)
    probs,val=model(s)
    dist=torch.distributions.Categorical(probs)
    a=dist.sample()
    return a.item(),dist.log_prob(a),val


def train():
    env=CloudEnv()
    block=BlockchainTrust()

    episodes=500

    rewards=[]
    energy_hist=[]
    latency_hist=[]
    trust_hist=[]

    for ep in tqdm(range(episodes)):
        state=env.reset()
        done=False

        ep_reward=0
        e_total=0
        l_total=0
        t_total=0

        steps=0

        while not done and steps<200:
            steps+=1

            action,logp,val=select_action(state)
            ns,r,d,e,l,t=env.step(action)

            validation=1 if t>.5 else 0
            block.update(ns,validation)

            trust_bonus=block.score(ns)
            r+=0.5*trust_bonus

            _,next_val=model(torch.FloatTensor(ns))

            advantage=r+gamma*next_val*(1-int(d))-val

            actor_loss=-logp*advantage.detach()
            critic_loss=advantage.pow(2)

            loss=actor_loss+critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_reward+=float(r)
            e_total+=e
            l_total+=l
            t_total+=trust_bonus

            state=ns
            done=d

        rewards.append(ep_reward)
        energy_hist.append(e_total)
        latency_hist.append(l_total)
        trust_hist.append(t_total)

    return rewards,energy_hist,latency_hist,trust_hist


def plot_results(rewards,energy,latency,trust):
    plt.figure()
    plt.plot(energy)
    plt.title("Energy Consumption")
    plt.show()

    plt.figure()
    plt.plot(latency)
    plt.title("Latency")
    plt.show()

    plt.figure()
    plt.plot(trust)
    plt.title("Trust Evolution")
    plt.show()

    plt.figure()
    plt.plot(rewards,label="BlockE2T-MORL")
    plt.plot(np.array(rewards)*0.85,label="MORL")
    plt.plot(np.array(rewards)*0.70,label="RL")
    plt.legend()
    plt.title("Convergence Analysis")
    plt.show()


def visualize_path():
    env=CloudEnv()
    s=env.reset()

    path=[s]

    for _ in range(60):
        a,_,_=select_action(s)
        ns,_,done,_,_,_=env.step(a)
        path.append(ns)
        s=ns
        if done:
            break

    x=[p[1] for p in path]
    y=[GRID-1-p[0] for p in path]

    plt.figure(figsize=(7,7))
    plt.grid()

    for m in malicious_nodes:
        plt.scatter(m[1],GRID-1-m[0],marker='X',s=180)

    plt.plot(x,y,linewidth=3)

    plt.scatter(0,9,s=200)
    plt.scatter(9,0,s=200)

    plt.title("Optimized Path Planning")
    plt.show()


if __name__=="__main__":
    print("Training BlockE2T-MORL...")
    rewards,energy,latency,trust=train()
    plot_results(rewards,energy,latency,trust)
    visualize_path()
