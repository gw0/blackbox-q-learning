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
from x.models import KerasModel
from x.memory import ExperienceReplay
from x.agent import DiscreteAgent

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
        self._state_shape = (1, self.n_features)
        self._state = np.zeros(self._state_shape)
        self._is_over = False
        self._prev_score = -float('inf')
        self._actions_log = []

    def observe(self):
        return self._state

    def update(self, action):
        self._actions_log.append(action[0])
        self._steps += 1
        self._prev_score = bbox.get_score()
        self._is_over = not bbox.do_action(action[0])
        self._state = bbox.get_state().reshape(self._state_shape)
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

            #if score == self._epoch_prev or score == self._epoch_max:
            self.train_steps += 0.1  # slowly increase steps
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


if __name__ == "__main__":
    # attach debugger
    def debugger(type, value, tb):
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        pdb.pm()
    import sys
    sys.excepthook = debugger

    # configuration
    epochs = 20000
    batch_size = 50  # retrieved from experience replay memory
    hidden_dim = 200  # hidden layer size
    memory_len = 500  # experience replay memory length
    epsilon = 0.1  # exploration rate epsilon
    gamma = 0.9  # discount rate gamma

    # agent environment
    agent_env = BlackBox(level="blackbox/levels/train_level.data")

    # learning model
    keras_model = Sequential()
    keras_model.add(Dense(hidden_dim, activation="softplus", input_dim=agent_env.n_features))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(hidden_dim, activation="softplus"))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(hidden_dim, activation="softplus"))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(agent_env.n_actions))
    agent_model = KerasModel(keras_model)

    # experience memory
    agent_mem = ExperienceReplay(memory_length=memory_len)

    # compile agent
    agent = DiscreteAgent(agent_model, agent_mem, epsilon=lambda *args: epsilon)
    # SGD optimizer + MSE cost + MAX policy = Q-learning as we know it
    agent.compile(optimizer=RMSprop(lr=0.001), loss='mse', policy_rule='max')

    # train agent
    agent.learn(agent_env, epoch=epochs, batch_size=batch_size, gamma=gamma)

    # save trained model and weights
    pre = "model-04-slow"
    with open(pre + ".json", 'w') as f:
        json.dump(keras_model.to_json(), f)
    keras_model.save_weights(pre + ".h5", overwrite=True)

    # test agent
    #agent.play(agent_env, epoch=100)

    bbox.finish(verbose=1)
