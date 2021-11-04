import gym
import collections
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def add_into_buffer(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        # return a batch of transitions
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            states.append(s)
            actions.append([a])
            rewards.append([r])
            next_states.append(s_prime)
            dones.append([done_mask])

        return  tf.convert_to_tensor(states, dtype = tf.float32), tf.convert_to_tensor(actions),\
                tf.convert_to_tensor(rewards), tf.convert_to_tensor(next_states, dtype=tf.float32), tf.convert_to_tensor(dones)
    
    def get_size(self):
        return len(self.buffer)

class DQN(Model):
    
    def __init__(self, action_n):
        super(DQN, self).__init__()
        self.h1 = Dense(128, activation='relu')
        self.h2 = Dense(128, activation='relu')
        self.q = Dense(action_n, activation='linear')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        q = self.q(x)
        return q


class DQN_agent():
    def __init__(self, env):
        self.BUFFER_SIZE = 20000
        self.BATCH_SIZE = 32
        self.EPSILON = 1.0
        self.EPSILON_DECAYING = 0.999
        self.GAMMA = 0.98

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
        
        self.optimizer = Adam(learning_rate=0.001)

    def select_action(self,state):
        if np.random.random() <= self.EPSILON:
            return self.env.action_space.sample()
        else:
            q_value = self.main_Qnetwork(tf.convert_to_tensor([state], dtype= tf.float32))
            return np.argmax(q_value.numpy())

    def anneal_eps(self):
        self.EPSILON *= self.EPSILON_DECAYING
        self.EPSILON = max(self.EPSILON,0.01)

    def update_target_network(self):
        self.target_Qnetwork.set_weights(self.main_Qnetwork.get_weights())
        
    def learn(self):
        s,a,r,s_prime,done_mask = self.replay_buffer.sample(self.BATCH_SIZE)
        
        target_qs = self.target_Qnetwork(s_prime)
        max_q = np.max(target_qs,axis=1, keepdims=True)
        td_targets = np.zeros(max_q.shape)
        for k in range(max_q.shape[0]):
            if done_mask[k]:
                td_targets[k] =r[k]
            else:
                td_targets[k] = r[k] + self.GAMMA * max_q[k]

        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(a,2) # 2 : # of action
            q = self.main_Qnetwork(s)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, self.main_Qnetwork.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_Qnetwork.trainable_variables))

    def train(self, num_episode):
        for n_epi in range(num_episode):
            time, episode_reward, done = 0, 0, False
            s = self.env.reset()
            done = False
            time = 0

            while not done:
                a = self.select_action(s)
                s_prime, r, done, _ = self.env.step(a)
            
                train_reward = r + time*0.01

                self.replay_buffer.add_into_buffer((s,a,train_reward, s_prime, done))
                s = s_prime

                time += 1

                if self.replay_buffer.get_size()>2000:
                    self.anneal_eps()
                    self.learn()
                s = s_prime
                episode_reward += r
                time += 1
            print('Episode: ', n_epi+1, 'Time: ', time, 'Reward: ', episode_reward)

        self.env.close()

def main():
    env = gym.make('CartPole-v1')
    agent = DQN_agent(env)
    agent.train(10000)

if __name__ == '__main__':
    main()