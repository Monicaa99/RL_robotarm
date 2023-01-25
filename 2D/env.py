import numpy as np
import pyglet

#Self representa la instancia de la clase. Al usar la palabra clave "self"
#podemos acceder a los atributos y metodos de la clase en Python.
#Vincula los atributos con los argumentos dados.

class ArmEnv(object):
    #Pyglet specific viewer, we could use others like pygame
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1] #DUDA
    #Especificamos un objetivo.La "l" indica el grosor
    goal = {'x': 100., 'y': 100., 'l': 40}
    state_dim = 9
    action_dim = 2

    def __init__(self):
        #Juntamos el tamanyo del brazo y el radio del brazo en una misma estructura

        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])#Numpy
        #Largo
        self.arm_info['l'] = 100        # 2 arms length
        #Radio
        self.arm_info['r'] = np.pi/6    # 2 angles information
        #Booleano que indica si se ha alcanzado el objetivo
        self.on_goal = 0

    def step(self, action):

        done = False
        action = np.clip(action, *self.action_bound) #DUDA
        #print(action)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    #Normalizamos los angulos

        (a1l, a2l) = self.arm_info['l']  # arm 1 and 2 largo
        (a1r, a2r) = self.arm_info['r']  # arm 1 and 2 radio
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        #a1xy_ es el punto entre los dos eslabones
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 acaba y a2 comienza (x1, y1)
        #Punto donde acaba el brazo 2
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 termina (x2, y2)
        # normalizar caracteristicas.Se utiliza 400 porque
        # se normalizan las distancias en funcion del tamanyo de la ventana
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        #Indicamos que queremos el dedo se acerque al objetivo
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)
        #Recompensa positiva si esta proximos.Negativa si estan alejados
        # Si el extremo esta entre un rango especifico
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                #Indica que estamos  cerca del objetivo
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True #Ha llegado al objetivo
        else:
            self.on_goal = 0

        # state (punto intermedio,finger,distancia de a1xy_+distancia finger,1 o 0(si alcanza o no el objetivo))
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        #print("SHAPE:",s.shape)
        return s, r, done

    def reset(self):

        self.goal['x'] = np.random.rand()*400.
        self.goal['y'] = np.random.rand()*400.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        #print("SHAPE:",s.shape)
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()


    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians

#Toma una estructura de datos arm_info y un objetivo y representar el estado en nuestro monitor.
class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        #print(self.goal_info)
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        #Representa el objetivo
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            #Hay que indicar los cuatro vertices del cuadrado.El objetivo es un cuadrado
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,  # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        #Representa el brazo 1
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        #Representa el brazo 2
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # Actualizamos el objetivo
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        #Suposicion de cuanto tienen que rotar las articulaciones
        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()

        #Rota los 4 vertices del brazo 1
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        #Rota los cuatro vertices del brazo 2
        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        #Actualizamos los nuevos vertices
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        #print("arm_1: ",np.concatenate((xy01, xy02, xy11, xy12)))
        #print("arm_2: ",np.concatenate((xy11_, xy12_, xy21, xy22)))

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y



if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
