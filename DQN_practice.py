import gym
import gym
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
import collections



class Qnet(tf.keras.Model):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(2, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q
    
class DQNAgent():
    def __init__(self):
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 2000

        self.memory = deque(maxlen=50000)

        self.current_model = Qnet()
        self.target_model = Qnet()
        self.optimizer = Adam(lr=self.learning_rate)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.set_weights(self.current_model.get_weights())
    
    def save_buffer(self,state,action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        return random.randrange(2) if (np.random.rand() <= self.epsilon) else np.argmax(self.current_model.predict(state))
    
    def train_model(self):
        if self.epsilon> self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.current_model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.current_model(states)
            one_hot_action = tf.one_hot(actions, 2)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    agent = DQNAgent()

    scores, episodes = [], []
    score_avg = 0
    num_episode = 10000
    
    for e in range(num_episode):
        done = False
        score = 0

        state = env.reset()
        state = state.reshape(1, -1)
        
        while not done:
            # env.render()

            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, -1)

            score += reward
            reward = 0.1 if not done or score == 500 else -1

            agent.save_buffer(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state
            if done and e % 20 == 0:
                agent.update_target_model()

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print('episode: {:3d} | score avg {:3.2f} | memory length: {:4d} | epsilon: {:.4f}'.format(e, score_avg, len(agent.memory), agent.epsilon))

                scores.append(score_avg)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel('episode')
                plt.ylabel('average score')
                plt.savefig('cartpole_graph.png')

                if score_avg  > 400:
                    agent.model.save_weights('./save_model/model', save_format='tf')
                    sys.exit()
