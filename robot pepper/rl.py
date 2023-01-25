# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()
#####################  hyper parameters  ####################
#Reforce learning

LR_A = 0.001    # learning rate for actor 0.001
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9  # reward discount 0.9
TAU = 0.01    # soft replacement 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session() #inicia un objeto de TensorFlow Graph en el que los tensores se procesan mediante operaciones
        #Contador acciones y contador estados inicializados a 0
        self.a_replace_counter, self.c_replace_counter = 0, 0
        #dimensiones de a s abound
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

        #placeholder:no se tiene que proporcionar un valor inicial. Se puede
        # especificar en tiempo de ejecuciun con el parametro feed_dict dentro de Session.run

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False) #a_next
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)#q_next

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement


        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_ #q=  reward + discount*q_next
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params) #Minimiza el TD-error Entrenar critic

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params) #Entrenar actor

        self.sess.run(tf.global_variables_initializer())

    #tf.Session.run(fetches,feed_dict=None,options=None,run_metadata=None)
    #Fetches: Single element of the grasp or a list of elements of your graphics
    #Esto puede devolver un tensor o una operacion.
    #Si es un tensor devuelve un valor evaluado
    #feed_dict: Sobreescribe el valor del tensor en el grafo
    #[,:] Hace referencia a filas.
    def choose_action(self, s):

        return self.sess.run(self.a, {self.S: s[None, :]})[0]



    def learn(self):

        # soft target replacement
        self.sess.run(self.soft_replace) #Ejecuta la red

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE) #Para  cada estado escogemos acciun aleatoria
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_): #Replay Buffer  el agente almacena las transiciones experimentadas, de modo que el agente pueda tomar muestras de ellas cuando aprenda.
        transition = np.hstack((s, a, [r], s_))
        #print("transition: ", transition)
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory

        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):

        #variable_scope: Devuelve un contexto para el ambito de las variables.
        with tf.variable_scope(scope):
            #Esta capa implementa la operaciun: outputs = activation(inputs * kernel + bias)
            #activation -> funciun de activaciun pasada como argumento de activaciun (si no es None)
            #kernel -> matriz de pesos creada por la capa
            # bias -> vector de bias creado por la capa (sulo si use_bias es True).

            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            #Escala la variable a dentro de [-1,1]
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        #crea la red, a partir del estado
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')
