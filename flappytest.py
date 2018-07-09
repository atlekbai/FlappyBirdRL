import random
from collections import deque

import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
import pylab
from ple import PLE
from ple.games.flappybird import *
from SumTree import SumTree


class FlappyBirdAgent:
    def __init__(self, state_len, action_len):
        self.state_len = state_len
        self.action_len = action_len
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.explored_states = SumTree(10000)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.sync_target_model()
        self.model_loaded = False

    def huber_loss(self, target, prediction):
        err = prediction - target

        cond = K.abs(err) < 2.0
        L2 = 0.5 * K.square(err)
        L1 = 2.0 * (K.abs(err) - 0.5 * 2.0)

        loss = tf.where(cond, L2, L1)

        return K.mean(loss)

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, activation="relu", input_shape=(8,)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(2, activation="linear"))
        # model.compile(optimizer=sgd(lr=learning_rate), loss="mse")
        model.compile(optimizer = Adam(lr = self.learning_rate), loss=self.huber_loss)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.explored_states.append((state, action, reward, next_state, done))

    def sync_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def pick_action(self, state, eps=None):
        eps = self.epsilon if eps is None else eps
        if numpy.random.rand() <= eps:
            return random.randrange(2)
        q_value = self.model.predict(state)
        return numpy.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, game_over):
        if self.epsilon == 1:
            game_over = True
         # Get TD-error and store it in brain
        expected_state = self.model.predict([state])
        old_val = expected_state[0][action]
        expected_state_val = self.target_model.predict([next_state])
        if game_over:
            expected_state[0][action] = reward
        else:
            expected_state[0][action] = reward + 0.98 * (
                numpy.amax(expected_state_val[0]))
        error = abs(old_val - expected_state[0][action])
        self.add(error, (state, action, reward, next_state, game_over))

    def sample(self, n):
        batch = []
        element = self.explored_states.total() / n
        for i in range(n):
            a = element * i
            b = element * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.explored_states.get(s)
            batch.append((idx, data))
        return batch

    def priority(self, loss):
        return (loss + 0.01) ** 0.9

    def update(self, idx, error):
        p = self.priority(error)
        self.explored_states.update(idx, p)

    def add(self, error, sample):
        p = self.priority(error)
        self.explored_states.add(p, sample)

    def replay(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay

        # Random sample extraction from explored_states in batch size
        mini_batch = self.sample(self.batch_size)

        errors = numpy.zeros(self.batch_size)
        states = numpy.zeros((self.batch_size, self.state_len))
        next_states = numpy.zeros((self.batch_size, self.state_len))
        actions, rewards, game_overs = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][1][0]
            actions.append(mini_batch[i][1][1])
            rewards.append(mini_batch[i][1][2])
            next_states[i] = mini_batch[i][1][3]
            game_overs.append(mini_batch[i][1][4])

        expected_output = self.model.predict(states)
        expected_output_val = self.target_model.predict(next_states)

        # Update expected_output using Bellman optimal equations
        for i in range(self.batch_size):
            old_val = expected_output[i][actions[i]]
            if game_overs[i]:
                expected_output[i][actions[i]] = rewards[i]
            else:
                expected_output[i][actions[i]] = rewards[i] + 0.98 * (
                    numpy.amax(expected_output_val[i]))
            errors[i] = abs(old_val - expected_output[i][actions[i]])

        for i in range(self.batch_size):
            sol = mini_batch[i][0]
            self.explored_states.update(sol, errors[i])

        self.model.fit(states, expected_output, batch_size=self.batch_size, epochs=1, verbose=0)

    def train(self, environment, game, self_model_path='model_backup.h5'):
          available_actions = [None, K_w]
          epoch = 1
          scores = []
          episodes = []
          index = 1
          while(True):
              game_over = False
              score = 0
              environment.reset_game()
              state = game.getGameState()
              states = []
              for key, value in state.iteritems():
                  states.append(value)
              state = numpy.reshape(numpy.array(states), [1, self.state_len])
              index += 1
              while not game_over:
                  epoch += 1
                  if self.model_loaded:
                      action = self.pick_action(state, 0)
                  else:
                      action = self.pick_action(state)
                  reward = environment.act(available_actions[action])

                  states = []
                  for key, value in game.getGameState().iteritems():
                      states.append(value)
                  next_state = numpy.array(states)
                  game_over = environment.game_over()
                  next_state = numpy.reshape(next_state, [1, self.state_len])

                  r = reward if not game_over else -10
                  self.append_sample(state, action, r, next_state, game_over)
                  if epoch >= 2500:
                      self.replay()

                  score += reward
                  state = next_state

                  if game_over:
                      self.sync_target_model()
                      print("epoch:", epoch, "  score:", (score + 5), "  eps:", self.epsilon)
                      scores.append(score+5)
                      episodes.append(index)
                      pylab.plot(episodes, scores, 'b')
                      pylab.savefig("graph.png")
                      if epoch % 20 == 0:
                          self.model.save("model_fb.h5")
                          self.model.save_weights("model_backup.h5")#weights
                          print("saved")

    def play(self, environment, game, self_model_path='model_backup.h5'):
          available_actions = [None, K_w]
          epoch = 1
          while(True):
              game_over = False
              environment.reset_game()
              score = 0
              state = game.getGameState()
              states = []
              for key, value in state.iteritems():
                  states.append(value)
              state = numpy.reshape(numpy.array(states), [1, self.state_len])

              while not game_over:
                 epoch += 1
                 action = self.pick_action(state, 0)
                 reward = environment.act(available_actions[action])

                 states = []
                 for key, value in game.getGameState().iteritems():
                     states.append(value)
                 next_state = numpy.array(states)
                 game_over = environment.game_over()
                 next_state = numpy.reshape(next_state, [1, self.state_len])

                 score += reward
                 state = next_state

                 if game_over:
                     print("epoch:", epoch, "score", score)


    def load_agent_experience(self, agent_weight_filepath):
        self.model.load_weights(agent_weight_filepath)
        self.sync_target_model()
        print("Model Loaded")
        return self.model


def play_flappy_bird(play_game=True, train_agent=False, agent_model_path='model_backup.h5'):
    game = FlappyBird()
    environment = PLE(game, fps=30, display_screen=True)
    # agent_explored_states = FlappyBirdAgent()
    action_len = 2
    states = []
    for key, value in game.getGameState().iteritems():
        states.append(value)
    print(states)
    state_len = len(states)

    agent_explored_states = FlappyBirdAgent(state_len, action_len)

    if os.path.exists(agent_model_path):
        agent_explored_states.load_agent_experience(agent_model_path)
        agent_explored_states.model_loaded = True
        print("WEights loaded")
    # environment.init()
    if train_agent:
        agent_explored_states.train(environment, game)
        print("Trained")
    if play_game:
        agent_explored_states.play(environment, game)
        print("Played")


play_flappy_bird(play_game=True, train_agent=True)
