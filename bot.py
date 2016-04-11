#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Deep Q-Learning for BlackBox Challenge (http://blackboxchallenge.com/).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import interface as bbox


def get_action_by_state(state, verbose=0):
    if verbose:
        for i in range(n_features):
            print ("state[%d] = %f" %  (i, state[i]))

        print ("score = {}, time = {}".format(bbox.get_score(), bbox.get_time()))

    action_to_do = 0
    return action_to_do


n_features = n_actions = max_time = -1

 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("blackbox/levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(verbose=False):
    has_next = 1
    
    prepare_bbox()
 
    while has_next:
        state = bbox.get_state()
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)
 
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=1)
 