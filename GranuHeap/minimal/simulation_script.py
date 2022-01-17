#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script.py
# License: GNU v3.0


import os

import numpy as np
import coexist

import plotly.graph_objs as go


# Generate a new LIGGGHTS simulation from the `granuheap_template.sim` template
nparticles = 2000

sliding = 0.6
rolling = 0.6
restitution = 0.3
cohesion = 0

density = 1580


# Directory to save results to
results_dir = "results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


# Load simulation template lines as a list[str] and modify input parameters
with open("granuheap_template.sim", "r") as f:
    sim_script = f.readlines()


# Simulation log path
sim_script[1] = f"log {results_dir}/granudrum.log\n"
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
sim_path = f"{results_dir}/granuheap.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)


# Run simulation up to given time (s)
line = "\n" + "-" * 80 + "\n"

print(line + "Pouring particles and letting them settle" + line)
sim.step_time(2.0)

print(line + "Lifting GranuHeap wall and letting particles settle" + line)
sim.execute_command("fix move all move/mesh mesh wall linear 0.0 0.0 0.005")

sim.step_time(4.0)


# Extract particle porperties as NumPy arrays
time = sim.time()
radii = sim.radii()
positions = sim.positions()
velocities = sim.velocities()

print("\n\n" + line)
print(f"Simulation time: {time} s\nParticle positions:\n{positions}")
print(line + "\n\n")


# Save results as efficient binary NPY-formatted files
np.save(f"{results_dir}/radii.npy", radii)
np.save(f"{results_dir}/positions.npy", positions)
np.save(f"{results_dir}/velocities.npy", velocities)


# Plot last particle configuration, colour-coded by their radii
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode="markers",
    marker_color=radii,
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
