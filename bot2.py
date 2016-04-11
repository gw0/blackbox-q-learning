#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Deep Q-Learning for BlackBox Challenge (http://blackboxchallenge.com/).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, RMSprop

from x.environment import Environment

import interface as bbox


class BlackBox(Environment):
    """BlackBox environment from the challenge."""

    def __init__(self, level, train_steps=1):
        self.level = level
        self.n_features = -1
        self.n_actions = -1
        self.max_time = -1

        self.train_steps = train_steps
        self._steps = 0
        self._state = None
        self._is_over = True
        self._prev_score = -float('inf')
        self._epoch_prev = -float('inf')
        self._epoch_max = -float('inf')
        self._actions_log = []

        self.reset()

    def reset(self):
        if bbox.is_level_loaded():
            bbox.reset_level()
        else:
            bbox.load_level(self.level, verbose=1)
            self.n_features = bbox.get_num_of_features()
            self.n_actions = bbox.get_num_of_actions()
            self.max_time = bbox.get_max_time()

        self._steps = 0
        self._state = np.zeros((1, self.n_features))
        self._is_over = False
        self._prev_score = -float('inf')
        self._actions_log = []

    def observe(self):
        return self._state

    def act(self, action):
        self._actions_log.append(action)
        self._steps += 1
        self._prev_score = bbox.get_score()
        self._is_over = not bbox.do_action(action)
        self._state = bbox.get_state().reshape((1, self.n_features))
        #print "\nupdate", self._prev_score, action, bbox.get_score(), self._is_over
        return self.state, self.reward(), self.is_over

    def reward(self):
        reward = bbox.get_score() - self._prev_score
        return reward

    @property
    def is_over(self):
        score = bbox.get_score()
        if score > self._epoch_max:
            self._epoch_max = score  # remember max score

        if self._steps >= self.train_steps:
            print "\nover (steps: {}/{}, score: {:.5}/{:.5})".format(self._steps, self.train_steps, score, self._epoch_max)
            print self._actions_log

            if score == self._epoch_prev or score == self._epoch_max:
                self.train_steps += 0.5  # increase steps after a while
            self._epoch_prev = score
            return True

        if score < -1. and score < -self._epoch_max / 2:
            print "\ndead (steps: {}/{}, score: {:.5}/{:.5})".format(self._steps, self.train_steps, score, self._epoch_max)
            print self._actions_log
            return True
        return self._is_over

    @property
    def state(self):
        return self._state

    @property
    def description(self):
        return "BlackBox challenge ({}, {}, {})".format(self.n_features, self.n_actions, self.max_time)


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    # attach debugger
    def debugger(type, value, tb):
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        pdb.pm()
    import sys
    sys.excepthook = debugger

    # parameters
    epsilon = .1  # exploration
    epoch = 1000
    max_memory = 500
    hidden_size = 200
    batch_size = 50
    grid_size = 10

    # agent environment
    env = BlackBox(level="blackbox/levels/train_level.data")
    num_actions = env.n_actions

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(env.n_features,), activation='softmax'))
    model.add(Dense(hidden_size, activation='softmax'))
    model.add(Dense(hidden_size, activation='softmax'))
    model.add(Dense(num_actions))
    #model.compile(SGD(lr=.2), "mse")
    model.compile(RMSprop(), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    for e in range(epoch):
        win_cnt = 0
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)[0]
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward >= 0.:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)[0]
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # save trained model and weights
    with open("model2.json", 'w') as f:
        json.dump(keras_model.to_json(), f)
    keras_model.save_weights("model2.h5", overwrite=True)

    # test agent
    #agent.play(agent_env, epoch=100)

    bbox.finish(verbose=1)

