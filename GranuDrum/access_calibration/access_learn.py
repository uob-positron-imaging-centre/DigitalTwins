#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.05.2021

import coexist

# Use ACCESS to learn a simulation's parameters
access = coexist.Access("simulation_script.py")
access.learn(num_solutions = 10, target_sigma = 0.1, random_seed = 12345)
