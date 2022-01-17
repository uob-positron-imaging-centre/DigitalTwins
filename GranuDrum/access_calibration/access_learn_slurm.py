#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn_slurm.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.05.2021


import coexist
from coexist.schedulers import SlurmScheduler


with open("bluebear_modules.sh") as f:
    commands = f.readlines()

scheduler = SlurmScheduler(
    "10:0:0",
    commands = commands,
    qos = "bbdefault",
    account = "windowcr-rt-royalsociety",
    constraint = "cascadelake",
)

# Use ACCESS to learn the simulation parameters
access = coexist.Access("simulation_script.py", scheduler)
access.learn(num_solutions = 100, target_sigma = 0.1, random_seed = 12345)
