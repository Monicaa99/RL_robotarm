# -*- coding: utf-8 -*-
"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG
import time


MAX_EPISODES = 900
MAX_EP_STEPS = 400
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim

a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset() #Estado
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()
            size = s.shape
            a = rl.choose_action(s)

            s_, r, done = env.step(a) #Estado siguiente, recompensa
            #print(s.shape)
            #print(a.shape)
            #print(r.shape)
            #print(s_.shape)
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()
    eval()


def eval():

    contador=0
    goal=0

    with open("comportamiento.txt",'w') as f:

        f.write("Número de episodios: "+ str(MAX_EPISODES) +"\n")
        f.write("Número de pasos por episodios: " +str(MAX_EP_STEPS)+"\n")

        while contador<100:


            rl.restore()
            env.render()
            env.viewer.set_vsync(True)
            s = env.reset()
            done=False
            objetivo=False
            start = time.time()
            end=0

            while  done==False and (end - start)<5:

                env.render()
                a = rl.choose_action(s)
                s, r, done = env.step(a)
                end = time.time()
                if(done):

                    goal+=1
                    objetivo=True

            if(objetivo):
                print (str(contador)+"___ done")

            else:
                print (str(contador)+"___ none")
            contador+=1
        f.write("Número de aciertos: " +str(goal)+"/"+str(contador)+"\n")
        f.close()


if ON_TRAIN:
    train()
else:

    eval()
