# encoding: utf-8
"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG
import pdb
#import keyboard
from naoqi import ALProxy #Comentar para entrenar
import time
import sys
import io
import math as m

MAX_EPISODES = 800
MAX_EP_STEPS = 700
ON_TRAIN = False
#ON_TRAIN = True
SHOW_VISU = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    global SHOW_VISU
    # start training
    for i in range(MAX_EPISODES):
        #print(i)
        s = env.reset() #Estado
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            #print(j)
            if keyboard.is_pressed('1') and SHOW_VISU is False:
                SHOW_VISU = True
                env.show_visu = True

            if keyboard.is_pressed('0') and SHOW_VISU is True:
                SHOW_VISU = False
                env.show_visu = False

            # env.render()

            a = rl.choose_action(s) #q1,q2,r
            #print("a: ", a)
            s_, r, done = env.step(a) #Estado siguiente, recompensa

            if s_ is None:
                #print("Salir")
                j==MAX_EP_STEPS
            else:

                rl.store_transition(s, a, r, s_)
                #print("r: ", r)
                ep_r += r
                #print("ep_r: ", ep_r)
                if rl.memory_full:
                    # start to learn once has fulfilled the memory
                    rl.learn()

                s = s_
                if done or j == MAX_EP_STEPS-1:
                    print('Ep: %i | %s | ep_r: %.1f | step: %i'% (i, '---' if not done else 'done', ep_r, j))
                    break
    rl.save()
    eval()

def eval():
    global SHOW_VISU
    contador=0
    goal=0
    SHOW_VISU=False
    PORT = 9559

    try:
        motionProxy = ALProxy("ALMotion", "127.0.0.1", PORT)
    except Exception,e:
        print "Could not create proxy to ALMotion"
        print "Error was: ",e
        sys.exit(1)

    with io.open("comportamiento.txt",'w',encoding = 'utf-8') as f:


        f.write(("Numero de episodios: "+ str(MAX_EPISODES) +"\n").decode('unicode_escape'))
        f.write(("Numero de pasos por episodios: " +str(MAX_EP_STEPS)+"\n").decode('unicode_escape'))

        while contador<100:

            rl.restore()
            env.render()
            env.show_visu = False
            s = env.reset()
            ep_r = 0.
            done=False
            objetivo=False
            start = time.time()
            end=0

            while done==False and (end - start)<10:

                env.render()
                a = rl.choose_action(s)
                s, r, done , q1, q2, q3= env.step_test(a)
                end = time.time()
                ep_r+=r
                #print(end - start)
                print("------TEST------------------")
                print(q1.item(), q2.item(), q3.item())

                #print("-----------------",type(a1rz.item()),type(0.1))

                if q3>2.0857:

                    q3= q3-2*m.pi
                    print(q1.item(), q2.item(),q3)

                #LWristYaw
                # Simple command for the HeadYaw joint at 10% max speed


                names            = ["LElbowYaw","LWristYaw","LShoulderRoll","LElbowRoll","LShoulderPitch"]
                angles           =[m.radians(0),m.radians(-21.2),q1.item(),q2.item(),q3.item()]
                times            =[0.1,0.1,0.1,0.1,0.1]
                motionProxy.setAngles(names,angles,times)


                if(done):

                    goal+=1
                    objetivo=True

            if(objetivo):
                print (str(contador)+"___ done")

            else:
                print (str(contador)+"___ none")
            contador+=1
        f.write(("Número de aciertos: " +str(goal)+"/"+str(contador)+"\n").decode('unicode_escape'))
        f.close()

if ON_TRAIN:
    train()
else:
    eval()
