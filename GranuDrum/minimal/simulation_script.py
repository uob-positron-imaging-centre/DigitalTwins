#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script.py
# License: GNU v3.0


import os

import numpy as np
import coexist

import plotly.graph_objs as go


# Generate a new LIGGGHTS simulation from the `granudrum_template.sim` template
rpm = 45
nparticles = 1000

sliding = 0.32
rolling = 0.0025
restitution = 0.3
cohesion = 0

density = 1580


# Directory to save results to
results_dir = "results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


# Load simulation template lines as a list[str] and modify input parameters
with open("granudrum_template.sim", "r") as f:
    sim_script = f.readlines()


# Simulation log path
sim_script[1] = f"log {results_dir}/granudrum.log\n"

sim_script[9] = f"variable rotationPeriod equal 60/{rpm}\n"
sim_script[10] = f"variable N equal {nparticles}\n"


# Parameter naming:
#    PP  = Particle-Particle
#    PW  = Particle-Wall (cylindrical hull of the GranuDrum)
#    PSW = Particle-Sidewall (circular sides of the GranuDrum)
sim_script[22] = f"variable fricPP equal {sliding}\n"
sim_script[23] = f"variable fricPW equal {sliding}\n"
sim_script[24] = f"variable fricPSW equal {sliding}\n"

sim_script[27] = f"variable fricRollPP equal {rolling}\n"
sim_script[28] = f"variable fricRollPW equal {rolling}\n"
sim_script[29] = f"variable fricRollPSW equal {rolling}\n"

sim_script[32] = f"variable corPP equal {restitution}\n"
sim_script[33] = f"variable corPW equal {restitution}\n"
sim_script[34] = f"variable corPSW equal {restitution}\n"

sim_script[37] = f"variable cohPP equal {cohesion}\n"
sim_script[38] = f"variable cohPW equal {cohesion}\n"
sim_script[39] = f"variable cohPSW equal {cohesion}\n"

sim_script[42] = f"variable dens equal {density}\n"


# Save the simulation template with the modified parameters
sim_path = f"{results_dir}/granudrum.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)


# Run simulation up to given time (s)
sim.step_to_time(0.1)


# Extract particle porperties as NumPy arrays
time = sim.time()
radii = sim.radii()
positions = sim.positions()
velocities = sim.velocities()

print("\n\n" + "-" * 80)
print(f"Simulation time: {time} s\nParticle positions:\n{positions}")
print("-" * 80 + "\n\n")


# Record particle properties at 120 Hz from t = 0.1 s up to t = 0.5 s
checkpoints = np.arange(0.1, 0.5, 1 / 120)

times = []
radii = []
positions = []
velocities = []

for t in checkpoints:
    sim.step_to_time(t)

    times.append(sim.time())
    radii.append(sim.radii())
    positions.append(sim.positions())
    velocities.append(sim.velocities())


# Save results as efficient binary NPY-formatted files
np.save(f"{results_dir}/times.npy", times)
np.save(f"{results_dir}/radii.npy", radii)
np.save(f"{results_dir}/positions.npy", positions)
np.save(f"{results_dir}/velocities.npy", velocities)


# Plot last particle configuration, colour-coded by their velocities
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=positions[-1][:, 0],
    y=positions[-1][:, 1],
    z=positions[-1][:, 2],
    mode="markers",
    marker_color=np.linalg.norm(velocities[-1], axis=1),
))

fig.update_layout(template="plotly_white")
fig.update_scenes(
    xaxis_range=[-0.042, 0.042],
    yaxis_range=[-0.042, 0.042],
    zaxis_range=[-0.042, 0.042],
    aspectmode="manual",
    aspectratio=dict(x=1, y=1, z=1),
)
fig.show()
