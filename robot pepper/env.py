import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pdb
import sys


#Self representa la instancia de la clase. Al usar la palabra clave "self"
#podemos acceder a los atributos y metodos de la clase en Python.
#Vincula los atributos con los argumentos dados.
class Viewer():

    def __init__(self, goal): #CREAMOS LOS PUNTOS INICIALES Y SE ALMACEN
        # vsync=False to not use the monitor FPS, we can speed up training
        plt.ion()
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.set_xlim3d(-200,200)
        self.ax.set_ylim3d(-200,200)
        self.ax.set_zlim3d(-200,200)
        self.ax.autoscale(enable=False)
        self.fig.canvas.set_window_title('Test')

    def render(self, a1xy, a1xy_, finger, goal):

        self.ax.clear()
        self.ax.set_xlim3d(-200,200)
        self.ax.set_ylim3d(-200,200)
        self.ax.set_zlim3d(-200,200)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.autoscale(enable=False)
        self.ax.plot3D([a1xy[0], a1xy_[0]], [a1xy[1], a1xy_[1]],[a1xy[2], a1xy_[2]])
        self.ax.plot3D([a1xy_[0], finger[0]], [a1xy_[1], finger[1]],[a1xy_[2], finger[2]])
        self.ax.scatter(goal['x'],goal['y'],goal['z'],s=50)

        plt.show()
        plt.pause(0.001) # '''





class ArmEnv(object):
    #Pyglet specific viewer, we could use others like pygame
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1] #DUDA



    #Especificamos un objetivo.La "l" indica el grosor
    goal = {'x': 50., 'y': 50., 'z': 0. ,'d': 50}
    center_coord = np.array([0, 0, 0])
    state_dim = 13
    action_dim = 6
    show_visu = False


    def __init__(self):
        #Juntamos el tamanyo del brazo y el radio del brazo en una misma estructura

        self.arm_info = np.zeros( 2, dtype=[('l', np.float32),('rx', np.float32), ('ry', np.float32), ('rz', np.float32)])#Numpy
        self.arm_info['l'] = 181.20,150
        #self.arm_info['l'] = 100   # 2 arms length


        self.arm_info['ry'][0]=np.random.randint(-2.0857,2.0857)#q3
        self.arm_info['ry'][1]=np.random.randint(-2.0857,2.0857)

        self.arm_info['rz'][0]=np.random.randint(0.0087,1.5620) #q1
        self.arm_info['rz'][1]=np.random.randint(-1.5620,-0.0087) #q2

        self.arm_info['rx']= 2 * np.pi * np.random.rand(2)

        self.on_goal = 0

        self.goal['x'] = np.random.randint(-200,200)
        self.goal['y'] = np.random.randint(50,300)
        self.goal['z'] = np.random.randint(-250,250)

        #print( self.arm_info)
        self.viewer = Viewer(self.goal)
    def inverse_kinematic(self):


        #Singularidad=False

        (l1,l2)=self.arm_info['l']
        #print(l1,l2)

        q2=-np.arccos((m.pow(self.goal['x'],2)+m.pow(self.goal['y'],2)-m.pow(l1,2)-m.pow(l2,2))/(2*l1*l2))
        q1=np.arctan2(self.goal['y'],self.goal['x'])-np.arcsin(l2*np.sin(q2)/m.sqrt(m.pow(self.goal['x'],2)+m.pow(self.goal['y'],2)))
        q3=m.radians(0)

        #print(l1,l2)
        d1=l1*np.cos(q3) #Proyeccion de a1l en el plano xz
        d2=(l2*np.cos(q3-m.pi/2)) #Proyeccion de a2l en el plano xz

        a1xy = self.center_coord
        proy_1=np.array([l1*np.cos(q1), l1*np.sin(q1),0]) + a1xy
        proy_2=np.array([l2*np.cos(q2+q1), l2*np.sin(q2+q1), 0]) + proy_1

        rotation_matrix=[[np.cos(q3),0,np.sin(q3) ],
                         [0,1,0],
                         [-np.sin(q3),0,np.cos(q3)]]

        a1xy_=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_1)))
        finger=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_2)))
    

        dist1 = [(self.goal['x'] - a1xy_[0])/400.0 , (self.goal['y'] - a1xy_[1])/400.0 ,(self.goal['z'] - a1xy_[2])/400.0]
        dist2 = [(self.goal['x'] - finger[0])/400.0 , (self.goal['y'] - finger[1])/400.0,(self.goal['z'] - finger[2])/400.0]

        dist2_ = [(self.goal['x'] - finger[0]) , (self.goal['y'] - finger[1]),(self.goal['z']- finger[2])]

        r=-np.linalg.norm(dist2)
        radius=np.linalg.norm(dist2_)

        if(0.0087<q1 and q1<1.5620) :
            pass
        else:
            return True #Singularidad

        if(-1.5620<q2 and q2<-0.0087) :
            pass
        else:
            return True #Singularidad


        return False




    def step(self, action):

        done = False

        action = np.clip(action, *self.action_bound)


        action=np.array_split(action,3)
        action_z=action[0] #Rotacion en eje z
        action_y=action[1] #Rotacion en eje y
        action_x=action[2] #Rotacion en eje x

        self.arm_info['rx'] += action_x * self.dt
        self.arm_info['rx'] %= np.pi * 2
        self.arm_info['ry'] += action_y * self.dt
        self.arm_info['ry'] %= np.pi * 2
        self.arm_info['rz'] += action_z * self.dt
        self.arm_info['rz'] %= np.pi * 2

        (a1l, a2l) = self.arm_info['l']  # arm 1 and 2 largo # radius, arm length
        (a1rx, a2rx) = self.arm_info['rx'] #Rotacion eje x
        (a1rz, a2rz) = self.arm_info['rz'] #Rotacion eje z
        (a1ry, a2ry) = self.arm_info['ry'] #Rotacion eje y

        #Singularidad=False
        #Comprobamos limites
        q1=a1rz
        q2=a2rz
        q3=a1ry
        l1=a1l
        l2=a2l



        d1=l1*np.cos(q3) #Proyeccion de a1l en el plano xz
        d2=(l2*np.cos(q3-m.pi/2)) #Proyeccion de a2l en el plano xz

        a1xy = self.center_coord
        proy_1=np.array([l1*np.cos(q1), l1*np.sin(q1),0]) + a1xy
        proy_2=np.array([l2*np.cos(q2+q1), l2*np.sin(q2+q1), 0]) + proy_1

        rotation_matrix=[[np.cos(q3),0,np.sin(q3) ],
                         [0,1,0],
                         [-np.sin(q3),0,np.cos(q3)]]

        a1xy_=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_1)))
        finger=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_2)))

        dist1 = [(self.goal['x'] - a1xy_[0])/400.0 , (self.goal['y'] - a1xy_[1])/400.0 ,(self.goal['z'] - a1xy_[2])/400.0]
        dist2 = [(self.goal['x'] - finger[0])/400.0 , (self.goal['y'] - finger[1])/400.0,(self.goal['z'] - finger[2])/400.0]

        dist2_ = [(self.goal['x'] - finger[0]) , (self.goal['y'] - finger[1]),(self.goal['z']- finger[2])]

        r=-np.linalg.norm(dist2)
        radius=np.linalg.norm(dist2_)


        q2=-2*m.pi+q2
        if(0.0087<q1 and q1<1.5620 ) :
            #pass
            r+=0
        else:
            r-=2

        if(-1.5620<q2 and q2<-0.0087) :
            #pass
            r+=0
        else:
            r-=2

        if(-2.0857<q3 and q3<2.0857) :
            #pass
            r+=0
        else:
            r-=2
            #Rango=False


        if self.show_visu is True:

            print(q1,q2,q3)
            #print(m.degrees(q1),m.degrees(q2),m.degrees(q3))
            self.viewer.render(a1xy, a1xy_, finger, self.goal)
            #plt.pause(10)
            print("distance to goal ",radius)
            #print("arminfo", self.arm_info)


        if radius  <  50 : #50
            r += 10 #1
            self.on_goal += 1 #10
            if self.on_goal > 20: #10
                done = True #Ha llegado al objetivo
        else:
            self.on_goal = 0


        # state (punto intermedio,finger,distancia de a1xy_+distancia finger,1 o 0(si alcanza o no el objetivo))
        s = np.concatenate((a1xy_/200.0, finger/200.0, dist1 + dist2, [1. if self.on_goal else 0.]))
        #print("s: ",s)


        return s, r, done


    def step_test(self, action):

        done = False

        action = np.clip(action, *self.action_bound)


        action=np.array_split(action,3)
        action_z=action[0] #Rotacion en eje z
        action_y=action[1] #Rotacion en eje y
        action_x=action[2] #Rotacion en eje x

        self.arm_info['rx'] += action_x * self.dt
        self.arm_info['rx'] %= np.pi * 2
        self.arm_info['ry'] += action_y * self.dt
        self.arm_info['ry'] %= np.pi * 2
        self.arm_info['rz'] += action_z * self.dt
        self.arm_info['rz'] %= np.pi * 2

        (a1l, a2l) = self.arm_info['l']  # arm 1 and 2 largo # radius, arm length
        (a1rx, a2rx) = self.arm_info['rx'] #Rotacion eje x
        (a1rz, a2rz) = self.arm_info['rz'] #Rotacion eje z
        (a1ry, a2ry) = self.arm_info['ry'] #Rotacion eje y




        #Singularidad=False
        #Comprobamos limites
        q1=a1rz
        q2=a2rz
        q3=a1ry
        l1=a1l
        l2=a2l



        d1=l1*np.cos(q3) #Proyeccion de a1l en el plano xz
        d2=(l2*np.cos(q3-m.pi/2)) #Proyeccion de a2l en el plano xz

        a1xy = self.center_coord
        proy_1=np.array([l1*np.cos(q1), l1*np.sin(q1),0]) + a1xy
        proy_2=np.array([l2*np.cos(q2+q1), l2*np.sin(q2+q1), 0]) + proy_1

        rotation_matrix=[[np.cos(q3),0,np.sin(q3) ],
                         [0,1,0],
                         [-np.sin(q3),0,np.cos(q3)]]

        a1xy_=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_1)))
        finger=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_2)))


        #print("a1xy_: ",a1xy_)
        #print("finger: ",finger)
        dist1 = [(self.goal['x'] - a1xy_[0])/400.0 , (self.goal['y'] - a1xy_[1])/400.0 ,(self.goal['z'] - a1xy_[2])/400.0]
        dist2 = [(self.goal['x'] - finger[0])/400.0 , (self.goal['y'] - finger[1])/400.0,(self.goal['z'] - finger[2])/400.0]

        dist2_ = [(self.goal['x'] - finger[0]) , (self.goal['y'] - finger[1]),(self.goal['z']- finger[2])]

        r=-np.linalg.norm(dist2)
        radius=np.linalg.norm(dist2_)


        q2=-2*m.pi+q2
        if(0.0087<q1 and q1<1.5620 ) :
            #pass
            r+=0
        else:
            #Rango=False
            #print("r-=1")
            r-=2

        if(-1.5620<q2 and q2<-0.0087) :
            #pass
            r+=0
        else:
            r-=2
            #Rango=False
            #print("r-=1")

        if(-2.0857<q3 and q3<2.0857) :
            #pass
            r+=0
        else:
            r-=2
            #Rango=False

        self.show_visu=True
        if self.show_visu is True:

            print(q1,q2,q3)
            #print(m.degrees(q1),m.degrees(q2),m.degrees(q3))
            self.viewer.render(a1xy, a1xy_, finger, self.goal)
            #plt.pause(10)
            print("distance to goal ",radius)
            #print("arminfo", self.arm_info)


        if radius  <  50 : #50
            r += 10 #1
            self.on_goal += 1 #10
            if self.on_goal > 20: #10
                done = True #Ha llegado al objetivo
        else:
            self.on_goal = 0


        # state (punto intermedio,finger,distancia de a1xy_+distancia finger,1 o 0(si alcanza o no el objetivo))
        s = np.concatenate((a1xy_/200.0, finger/200.0, dist1 + dist2, [1. if self.on_goal else 0.]))
        #print("s: ",s)

        return s, r, done,q1,q2,q3



    def reset(self):

        self.goal['x'] = np.random.randint(-50,300)
        self.goal['y'] = np.random.randint(-50,300)
        self.goal['z'] = np.random.randint(-200,200)

        self.arm_info['rx']=0

        self.arm_info['ry'][0]=np.radians(90)#q3
        self.arm_info['ry'][1]=np.radians(90)

        self.arm_info['rz'][0]=np.radians(90)
        self.arm_info['rz'][1]=np.radians(0) #q2

        repeat=True
        Singularidad=False

        while repeat:

            self.goal['x'] = np.random.randint(-50,300)
            self.goal['y'] = np.random.randint(-50,300)
            self.goal['z'] = np.random.randint(-200,200)

            Singularidad=self.inverse_kinematic()
            repeat=Singularidad

        self.on_goal = 0

        (a1l, a2l) = self.arm_info['l']  # arm 1 and 2 largo
        (a1rz, a2rz) = self.arm_info['rz']
        (a1ry, a2ry) = self.arm_info['ry'] #Para ambos brazos sera igual
        (a1rx, a2rx) = self.arm_info['rx']#Para ambos brazos sera igual


        q1=a1rz
        q2=a2rz
        q3=a1ry
        l1=a1l
        l2=a2l

        d1=l1*np.cos(q3) #Proyeccion de a1l en el plano xz
        d2=(l2*np.cos(q3-m.pi/2)) #Proyeccion de a2l en el plano xz

        a1xy = self.center_coord
        proy_1=np.array([l1*np.cos(q1), l1*np.sin(q1),0]) + a1xy
        proy_2=np.array([l2*np.cos(q2+q1), l2*np.sin(q2+q1), 0]) + proy_1

        rotation_matrix=[[np.cos(q3),0,np.sin(q3) ],
                         [0,1,0],
                         [-np.sin(q3),0,np.cos(q3)]]

        a1xy_=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_1)))
        finger=np.transpose(np.matmul(rotation_matrix,np.transpose(proy_2)))


        dist1 = [(self.goal['x'] - a1xy_[0])/400.0 , (self.goal['y'] - a1xy_[1])/400.0 ,(self.goal['z'] - a1xy_[2])/400.0]
        dist2 = [(self.goal['x'] - finger[0])/400.0 , (self.goal['y'] - finger[1])/400.0,(self.goal['z'] - finger[2])/400.0]


        # state
        s = np.concatenate((a1xy_, finger, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            print("No hay creado un view")
            self.viewer = Viewer(self.arm_info, self.goal)

        #self.show_visu = True

        self.viewer.ax.set_xlim3d(-200,200)
        self.viewer.ax.set_ylim3d(-200,200)
        self.viewer.ax.set_zlim3d(-200,200)
        self.viewer.ax.set_xlabel('X axis')
        self.viewer.ax.set_ylabel('Y axis')
        self.viewer.ax.set_zlabel('Z axis')

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians

#Toma una estructura de datos arm_info y un objetivo y representar el estado en nuestro monitor.



if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
