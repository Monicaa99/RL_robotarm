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
import sys, os
import io
import math as m
import cv2
import vision_definitions
import numpy as np
from PIL import Image


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

    #motionProxy = ALProxy("ALMotion", "127.0.0.1", PORT) #Robot Real

    try:
        #motionProxy = ALProxy("ALMotion", "172.18.33.122", PORT) #Robot Real
        #motionProxy = ALProxy("ALMotion", "127.0.0.1", PORT) #Robot simulado
        motionProxy = ALProxy("ALMotion", "192.168.1.81", PORT)
        camProxy=ALProxy("ALVideoDevice","192.168.1.81",PORT)

    except Exception,e:
        print "Could not create proxy to ALMotion"
        print "Error was: ",e
        sys.exit(1)


    while(True):

        rl.restore()
        env.render()
        env.show_visu = False
        file_exist=False

        if os.path.exists("/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt") is True:

			os.remove("/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt")

        '''
        #WEBCAM
        cap = cv2.VideoCapture(-1)
        ret, frame = cap.read()
        cv2.imwrite(r"/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg", frame)
        #cv2.imshow('image', frame)
        cap.release()
        '''
        #Pepper Camera

        # Get the service ALVideoDevice.
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        print("GETTING IMAGE")
        time.sleep(3)
        videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
        Pepper_Image = camProxy.getImageRemote(videoClient)

        camProxy.unsubscribe(videoClient)


        # Now we work with the image returned and save it as a PNG  using ImageDraw
        # package.

        # Get the image size and pixel array.
        imageWidth = Pepper_Image[0]
        imageHeight = Pepper_Image[1]
        array = Pepper_Image[6]
        image_string = str(bytearray(array))

        # Create a PIL Image from our pixel array.
        im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

        # Save the image.
        im.save("/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg")

        time.sleep(5)
        user_goal=""

        while file_exist==False:
            print("Waiting goal...")
            if os.path.exists("/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt") is True:
                file_exist=True

        time.sleep(5)
        file = open("/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt", 'r')

        for line in file:

            user_goal=line

        file.close()

        if os.path.exists("/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg") is True:

			os.remove("/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg")

        #print(user_goal)
        user_goal=user_goal.split()
        user_goal_X=float(user_goal[0])
        user_goal_Y=float(user_goal[1])
        user_goal_Z=float(user_goal[2])


        env.goal['x']=user_goal_X
        env.goal['y']=user_goal_Y
        env.goal['z']=user_goal_Z


        s = env.reset_test()

        if s.all()==None:
            print("Establecer pose valida")

        else:

            ep_r = 0.
            done=False
            objetivo=False
            start = time.time()
            end=0

            #Capturar imagen
            #Leer guardar en carpeta de intercambio
            #Esperar fichero
            while done==False and (end - start)<15 :

                env.render()
                a = rl.choose_action(s)
                s, r, done , q1, q2, q3= env.step_test(a)
                end = time.time()

                if q3>2.0857:

                    q3= q3-2*m.pi
                    #print(q1.item(), q2.item(),q3)


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



if ON_TRAIN:
    train()
else:
    eval()
