import gym
import collections
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def append(self,transition):
        self.buffer.append(transition)

    def sample(self,batch_size):
        size = batch_size if len(self.buffer)> batch_size else len(self.buffer)
        return random.sample(self.buffer, size)

    def clear(self):
        self.buffer.clear()

    def get_size(self):
        return len(self.buffer)

class DQN(Model):
    
    def __init__(self, action_n):
        super(DQN, self).__init__()
        self.h1 = Dense(32, activation='relu')
        self.h2 = Dense(16, activation='relu')
        self.q = Dense(action_n, activation='linear')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        q = self.q(x)
        return q


class DQN_agent():
    def __init__(self, env):
        self.BUFFER_SIZE = 50000
        self.BATCH_SIZE = 32
        self.EPSILON = 1.0
        self.EPSILON_DECAYING = 0.995
        self.GAMMA = 0.95

        self.action_n = env.action_space.n #2
        self.state_dim = env.observation_space.shape[0] #4
        
        self.env = env
        self.main_Qnetwork = DQN(self.action_n)
        self.target_Qnetwork = DQN(self.action_n)

        self.main_Qnetwork.build(input_shape=(None, self.state_dim))
        self.target_Qnetwork.build(input_shape=(None, self.state_dim))

        self.main_Qnetwork.summary()

        self.target_Qnetwork.set_weights(self.target_Qnetwork.get_weights())
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE)
        
        self.optimizer = Adam(learning_rate=0.0005)
        self.loss_object = MeanSquaredError()

    def select_action(self,state):
        if np.random.random() <= self.EPSILON:
            return self.env.action_space.sample()
        else:
            q_value = self.main_Qnetwork(tf.convert_to_tensor([state], dtype= tf.float32))
            return np.argmax(q_value.numpy())

    def update_target_network(self):
        self.target_Qnetwork.set_weights(self.main_Qnetwork.get_weights())
    
    @tf.function
    def learn(self,states, actions, rewards, next_states, dones):
        td_target = rewards + (1-dones) * self.GAMMA * tf.reduce_max(self.target_Qnetwork(next_states), axis=1, keepdims=True)
        
        with tf.GradientTape() as tape:
            q = self.main_Qnetwork(states)
            one_hot_actions = tf.one_hot(tf.cast(tf.reshape(actions,[-1]),tf.int32),2) 
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)   
            loss = self.loss_object(td_target,q_values)

        grads = tape.gradient(loss, self.main_Qnetwork.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_Qnetwork.trainable_variables))

    def train(self, num_episode):
        score_avg = 0
        for n_epi in range(num_episode):
            self.EPSILON *= self.EPSILON_DECAYING
            self.EPSILON = max(self.EPSILON,0.01)
            done, score = False, 0
            state = self.env.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                transition = (state, action, reward/100.0, next_state, done)
                self.replay_buffer.append(transition)

                score += reward
                
                if self.replay_buffer.get_size() >= 2000:
                    transitions= self.replay_buffer.sample(self.BATCH_SIZE)
                    self.learn(*map(lambda x: np.vstack(x).astype('float32'), np.transpose(transitions)))
                state = next_state

                if done and n_epi % 20 == 0:
                    self.update_target_network()
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    print('episode: {:3d} | score avg {:3.2f} | memory length: {:4d} | epsilon: {:.4f}'.format(n_epi, score_avg, self.replay_buffer.get_size(), self.EPSILON))


def main():
    env = gym.make('CartPole-v1')
    agent = DQN_agent(env)
    agent.train(10000)

if __name__ == '__main__':
    main()