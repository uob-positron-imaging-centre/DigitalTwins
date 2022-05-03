#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script_template.py
# License: GNU v3.0


import os

import numpy as np
import coexist


# Generate a new LIGGGHTS simulation from the `granuflow_template.sim` template
nparticles = 130000

sliding = 0.3197
rolling = 0.00248
restitution = 0.3
cohesion = 0

density = 1580

orifice_size = 2

# Directory to save results to
simulation_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
results_dir = f"{simulation_dir}/results/Orifice_{orifice_size:02d}mm"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# Load simulation template lines as a list[str] and modify input parameters
with open(f"{simulation_dir}/Templates/granuflow_template.sim", "r") as f:
    sim_script = f.readlines()


# Simulation log path
sim_script[1] = f"log {results_dir}/granuflow.log\n"
sim_script[10] = f"variable N equal {nparticles}\n"


# Parameter naming:
#    PP  = Particle-Particle
#    PW  = Particle-Wall (cylindrical hull of the granuflow)
#    PSW = Particle-Sidewall (Bottom plate)
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

sim_script[134] = f"fix	plate	all mesh/surface file mesh/Plate{orifice_size}mm.stl	type 2  scale 0.001\n"


# Save the simulation template with the modified parameters
sim_path = f"{simulation_dir}/simulation_files/granuflow_{orifice_size}.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)


# Run simulation up to given time (s)
line = "\n" + "-" * 80 + "\n"

# Inserting Particles

print(line + "Filling GranuFlow cylinder" + line)
sim.step_to_time(2.0)  # 2 seconds

# Letting Particles settle

print(line + "Letting Particles Settle" + line)
sim.step_to_time(3.0)  # 1 second

# Calculate starting mass in the system

radii = sim.radii()
radii = radii[~np.isnan(radii)]  # Remove any nan values from array.

volume = (4 / 3) * np.pi * (radii ** 3)
mass_array = volume * density
start_mass = np.sum(mass_array)  # Mass after all particles inserted


print(line + "Moving GranuFlow plate and letting particles flow" + line)

times = []
positions = []
radii = []
velocities = []
mass = []

sim.execute_command("fix MovePlate all move/mesh mesh plate linear -0.05 0. 0.")  # Move GranuFlow plate.

checkpoints_open = np.arange(3.0, 4.0, 1/100)  # Open plate over 1 second (Saving data 100 times a second!)

for t in checkpoints_open:
    sim.step_to_time(t)

    radii = sim.radii()
    radii = radii[~np.isnan(radii)]  # Remove any nan values from array.

    volume = (4/3)*np.pi*(radii**3)  # Calculate the current volume of all the particles in the system
    mass_array = volume*density  # Calculate mass of all partiles currently in the system
    mass_at_t = np.sum(mass_array)  # Sum all array elements to got one value of mass at time t.
    mass_at_t = start_mass - mass_at_t  # Find mass that has left the system.

    mass.append(mass_at_t)  # Append current mass as time t to mass array

    # times.append(sim.time())
    # radii.append(sim.radii())
    # positions.append(sim.positions())
    # velocities.append(sim.velocities())

sim.execute_command("unfix MovePlate")

print(line + "Allowing Particles to Flow" + line)

start_time = 4.0
end_time = 8.0

checkpoints = np.arange(start_time, end_time, 1 / 100)  # Allowing particles to flow for 4 seconds after plate
# is fully moved

for t in checkpoints:
    sim.step_to_time(t)

    radii = sim.radii()
    radii = radii[~np.isnan(radii)]  # Remove any nan values from array.

    volume = (4/3)*np.pi*(radii**3)
    mass_array = volume*density
    mass_at_t = np.sum(mass_array)
    mass_at_t = start_mass - mass_at_t

    mass.append(mass_at_t)

    # times.append(sim.time())
    # radii.append(sim.radii())
    # positions.append(sim.positions())
    # velocities.append(sim.velocities())

# Save results as efficient binary NPY-formatted files
# np.save(f"{results_dir}/times_{orifice_size}mm.npy", times)
# np.save(f"{results_dir}/radii_{orifice_size}mm.npy", radii)
# np.save(f"{results_dir}/positions_{orifice_size}mm.npy", positions)
# np.save(f"{results_dir}/velocities_{orifice_size}mm.npy", velocities)
np.save(f"{results_dir}/mass_{orifice_size}mm.npy", mass)
