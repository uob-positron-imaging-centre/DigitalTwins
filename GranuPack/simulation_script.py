#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script.py
# License: GNU v3.0


import os
import math
import numpy as np
import coexist


def calc_bulk_density(positions, radii):

    positions = positions[~np.isnan(positions).any(axis=1)]  # Remove nans
    radii = radii[~np.isnan(radii)]  # Remove nans
    zpositions_radii_add = np.add(positions[:, 2], radii)
    zpositions_radii_subtract = np.subtract(positions[:, 2], radii)
    highest_particles_indicies = np.argpartition(zpositions_radii_add, -10)[-10:]
    bottom = np.nanmin(zpositions_radii_subtract)
    highest_particle_positions = zpositions_radii_add[highest_particles_indicies]
    top = np.mean(highest_particle_positions)
    length = top - bottom
    total_powder_volume = length*math.pi*tube_internal_radius**2
    calculated_bulk_density = start_mass/total_powder_volume

    return calculated_bulk_density


# Generate a new LIGGGHTS simulation from the `granupack_template.sim` template
# User dependent values
nparticles = 20000
drop_distance = 1/1000  # m
no_of_taps = 150
gravity = 9.81
rest_time = 0.2  # Time for particles to settle between drops
tinitial = 1  # Initial time to let particles settle during/after insertion
save_bulk_density = True  # Save numpy arrays of bulk density
save_all_particle_data = True  # Save particle data. Positions, radii.

# Particle Properties
sliding = 0.3197
rolling = 0.00248
restitution = 0.3
cohesion = 0
density = 1580

# Geometry Constants
tube_internal_radius = 0.026/2  # m

# Directory to save results to
simulation_dir = os.path.normpath(os.getcwd())

working_dir = f"{simulation_dir}/simulation_files"
if not os.path.isdir(working_dir):
    os.makedirs(working_dir)

results_dir = f"{working_dir}/results"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# Load simulation template lines as a list[str] and modify input parameters
with open(f"{simulation_dir}/granupack_template.sim", "r") as f:
    sim_script = f.readlines()

# Simulation log path
sim_script[1] = f"log {working_dir}/granupack.log\n"
sim_script[10] = f"variable N equal {nparticles}\n"

# Parameter naming:
#    PP  = Particle-Particle
#    PW  = Particle-Wall (cylindrical hull of the GranuPack)
sim_script[21] = f"variable fricPP equal {sliding}\n"
sim_script[22] = f"variable fricPW equal {sliding}\n"

sim_script[24] = f"variable fricRollPP equal {rolling}\n"
sim_script[25] = f"variable fricRollPW equal {rolling}\n"

sim_script[27] = f"variable corPP equal {restitution}\n"
sim_script[28] = f"variable corPW equal {restitution}\n"

sim_script[30] = f"variable cohPP equal {cohesion}\n"
sim_script[31] = f"variable cohPW equal {cohesion}\n"

sim_script[33] = f"variable dens equal {density}\n"

# Save the simulation template with the modified parameters
sim_path = f"{working_dir}/granupack.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)


# Run simulation up to given time (s)
line = "\n" + "-" * 80 + "\n"
droptime = np.sqrt((2*drop_distance)/gravity)  # sec

# Inserting Particles

print(line + "Filling GranuPack cylinder and letting particles settle." + line)
sim.step_to_time(tinitial)  # 2 seconds

# Calculate initial mass of particles

initial_radii = sim.radii()
initial_radii = initial_radii[~np.isnan(initial_radii)]  # Remove any nan values from array.

volume = (4 / 3) * np.pi * (initial_radii ** 3)
mass_array = volume * density
start_mass = np.sum(mass_array)  # Mass after all particles inserted

print(line + "Starting tapping" + line)

times_array = []
positions_array = []
radii_array = []
velocities_array = []
mass = []
bulk_density = []

# Loop to commence tapping
tlast = tinitial
discretize = 20
tdrop = 0
for d in range(no_of_taps):
    checkpoints = np.linspace(tlast, tlast+droptime, discretize+1)

    print("\n\n" + "-" * 80)
    print(f"Tap number: {d}")
    print("-" * 80 + "\n\n")

    for t, tdrop in enumerate(checkpoints):
        if t == 0:
            continue
        drop_velocity = gravity*(tdrop-tlast)
        sim.execute_command(f"fix DropTube all move/mesh mesh tube linear 0. 0. -{drop_velocity}")  # Drop tube.
        sim.step_to_time(tdrop)
        sim.execute_command("unfix DropTube")

    # Settle time
    t_settle = tdrop + rest_time
    sim.step_to_time(t_settle)
    tlast = t_settle

    # Calculate bulk density
    bulk_density_at_d = calc_bulk_density(sim.positions(), sim.radii())

    # Append bulk density
    bulk_density.append(bulk_density_at_d)

    # Append Particle Data
    times_array.append(sim.time())
    radii_array.append(sim.radii())
    positions_array.append(sim.positions())
    velocities_array.append(sim.velocities())

# Save results as efficient binary NPY-formatted files
if save_bulk_density is True:
    np.save(f"{results_dir}/bulk_density.npy", bulk_density)

if save_all_particle_data is True:
    np.save(f"{results_dir}/radii.npy", radii_array)
    np.save(f"{results_dir}/positions.npy", positions_array)
