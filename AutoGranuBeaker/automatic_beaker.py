#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script.py
# License: GNU v3.0
# Author : Dominik Werner

# This script generates a Beaker geometry based on the provided particle size distribution and runs a LIGGGHTS simulation.
# provide the csv file in the command line arguments.
# the --default or -d flag, the default 50 ml beaker will be used.
# the --unit or -u flag, the unit of the csv file can be specified. Valid units are mm, cm and m. Default is m.
# the --show flag, the generated beaker will be shown in a 3D viewer.
# the -o the psd file will be overwritten. Default is to raise an error if the file already exists.
# the -i or --insert_multiplier flag, the the insertion rate will be multiplied by the provided value. Default is 1.
# the -v or --vel flag, the velocity of the particles will be inserted. Default is 0.05 m/s.
# the --insertion_length flag, to add to the insertion length. Default is 0.1 m. This defines the region particles will initially included in the simulation.
# TODO: Figure out how to use parameters? Maybe a parameter file?

# make the beaker x times bigger than the biggest particle
import sys
import coexist
import numpy as np
import os
import sympy
import pint
np.random.seed(0)
BEAKER_FACTOR = 20


def reduce_kin_energy(sim, threshold=1e-14):
    sim.step_time(0.1)
    radii = sim.radii()
    vel = sim.velocities()
    kin_energy = 0.5 * np.sum(4/3 * np.pi * radii **
                              3 * np.linalg.norm(vel, axis=1)**2)
    print("Kinetic Energy = ", kin_energy)
    while kin_energy > threshold:
        sim.step_time(0.1)
        radii = sim.radii()
        vel = sim.velocities()
        kin_energy = 0.5 * np.sum(4/3 * np.pi * radii**3 * vel**2)
        print("Kinetic Energy = ", kin_energy)


# Parse the command line arguments and check if the provided csv file is valid
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
    if not csv_file.endswith(".csv"):
        raise ValueError("Please provide a csv file not a " +
                         csv_file.split(".")[-1])
if "--unit" in sys.argv or "-u" in sys.argv:
    try:
        unit = sys.argv[sys.argv.index("--unit") + 1]
        if unit not in ["µm", "mm", "cm", "m"]:
            raise ValueError("Please provide a valid unit")
    except IndexError:
        raise ValueError("Please provide a valid unit")
else:
    unit = "m"
convert_factor = pint.UnitRegistry().convert(1, unit, "m")

print("Unit of CSV: " + unit)
if not ("--default" in sys.argv or "-d" in sys.argv):
    gen_mesh = True
    try:
        import gmsh
        cad = gmsh.model.occ
    except Exception as e:
        print("Error: Gmsh is not installed. Please install Gmsh and try again.")
        print(e)
        exit()
else:
    gen_mesh = False
    print("Using default 50 ml beaker")

# show the generated beaker
if "--show" in sys.argv or "-s" in sys.argv:
    show = True
else:
    show = False

# Granubeaker mesh generation class


class GranubeakerMesh:

    def __init__(self, diameter, height):
        self.diameter = diameter
        self.height = height
        gmsh.initialize()
        #gmsh.option.setNumber("General.NumThreads", 8)
        #gmsh.option.setNumber("Geometry.OCCParallel", 8)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)

        gmsh.model.add("GranuBeaker")
        self.cylinder_tags = cad.addCylinder(
            0, 0, 0,
            0, 0, height,
            diameter/2
        )
        cad.synchronize()
        # meshing of the volume
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
        gmsh.model.mesh.generate(2)

    def save(self, path):
        gmsh.write(path)
        gmsh.finalize()
        # also write a insertion file
        gmsh.initialize()
        gmsh.model.add("GranuBeakerInit")
        curvetag = cad.addCircle(0, 0, self.height*0.75, self.diameter/2)
        wiretag = cad.addWire([curvetag])
        surfacetag = cad.addPlaneSurface([wiretag])
        cad.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(path.split("/")[0]+"/granubeaker_init.stl")
        # gmsh.fltk.run()
        gmsh.finalize()


# read csv data but check if there is a header
csv_data = np.genfromtxt(csv_file, delimiter=",")


print(csv_data.shape)
if csv_data.shape == (2,):
    # only one column
    particle_diameters = np.asarray([csv_data[0] * convert_factor])
    particle_fractions = np.asarray([csv_data[1]])
else:
    # remove header
    csv_data = csv_data[np.isfinite(csv_data[:, 0])]
    particle_diameters = csv_data[:, 0] * convert_factor
    particle_fractions = csv_data[:, 1]

particle_fractions /= np.sum(particle_fractions)
weighted_particle_volumes = (4/3) * np.pi * \
    (particle_diameters/2)**3 * particle_fractions
average_particle_volume = np.median(weighted_particle_volumes)
average_particle_diameter = np.average(
    particle_diameters, weights=particle_fractions)

unit = "m"
print("\nBiggest particle diameter: " +
      str(max(particle_diameters)) + " " + unit)
if gen_mesh:
    print("Generating mesh...")
    # max(particle_diameters)
    diameter = BEAKER_FACTOR * average_particle_diameter
    height = diameter
    print("Diameter of Beaker: " + str(diameter) + " " + unit)
    volume_beaker = np.pi * (diameter / 2)**2 * height
    print("Volume of Beaker: " + str(volume_beaker) + " " + unit + "^3")
    granubeaker = GranubeakerMesh(diameter, height)
    if show:
        gmsh.fltk.run()
    granubeaker.save("mesh/granubeaker.stl")
else:
    # using 50 ml beaker
    if not os.path.isfile("mesh/beaker_50ml.stl"):
        raise FileNotFoundError("mesh/beaker_50ml.stl not found")
    os.system("cp mesh/beaker_50ml.stl mesh/granubeaker.stl")
    # copy "mesh/ins_mesh_50ml.stl" to "mesh/granubeaker_init.stl"
    if not os.path.isfile("mesh/ins_mesh_50ml.stl"):
        raise FileNotFoundError("mesh/ins_mesh_50ml.stl not found")
    os.system("cp mesh/ins_mesh_50ml.stl mesh/granubeaker_init.stl")


################ generate LIGGGGHTS PSD file ################
filename = "psd.sim"
string = []
lines = []
for i, particle in enumerate(zip(particle_diameters, particle_fractions)):
    prime = sympy.prime(1003253+i)
    r, psd = particle
    r /= 2
    lines.append(f"variable r{i}             equal {r}\n")
    lines.append(f"variable p{i}             equal {psd}\n")

    lines.append(f"fix pts{i} all particletemplate/sphere {prime} atom_type 1 density constant " +
                 "${dens} radius constant ${r"+f"{i}"+"} \n")
    string.append(" pts"+str(i)+" ${p"+str(i)+"} ")
lines.append(
    f"fix pdd all particledistribution/discrete/numberbased 32452843 {i+1}"+"".join(string) + "\n")
if not "-o" in sys.argv:
    if os.path.isfile(filename):
        raise FileExistsError(
            "File "+filename+" already exists. Please delete it or use the -o flag to overwrite it.")
with open(filename, "w") as f:
    for line in lines:
        f.write(line)

# Generate a new LIGGGHTS simulation from the `cylinder_template.sim` template

# DEM Parameters

sliding = 0.3197
rolling = 0.00248
restitution = 0.3
cohesion = 0
density = 1580.0

# Python script parameters

# Value that sets the sensitivity of the deletion steps. (Continue to next step if the number of particles deleteed is below this amount)
particle_tolerance = 10

# Directory to save results to
results_dir = "results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


# Load simulation template lines as a list[str] and modify input parameters
with open("granubeaker_template.sim", "r") as f:
    sim_script = f.readlines()


# Simulation log path
sim_script[1] = f"log {results_dir}/granubeaker.log\n"
#sim_script[10] = f"variable N equal {nparticles}\n"


# Parameter naming:
#    PP  = Particle-Particle
#    PW  = Particle-Wall (cylindrical hull of the granubeaker)

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

if "--insertion_length" in sys.argv:
    insertion_length = float(sys.argv[sys.argv.index(
        "--insertion_length")+1])
else:
    insertion_length = 0.0


# define domain of the system
xmin = -diameter / 2 * 1.1
xmax = diameter / 2 * 1.1
ymin = -diameter / 2 * 1.1
ymax = diameter / 2 * 1.1
zmin = -0.01 * 1.1
zmax = height * 1.1 + insertion_length
sim_script[57] = f"region domain block {xmin} {xmax} {ymin} {ymax} {zmin} {zmax} units box\n"

# set particle skin distance
sim_script[62] = f"neighbor	    {np.max(particle_diameters)/2} bin\n"

# Save the simulation template with the modified parameters
sim_path = f"granubeaker.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)


# empty simulation, so just bare bones
# we now fill the system with particles until half of the volume is filled

# estimate the number of particles based on the volume of the beaker and the psd


# calculate necessery parameters for insertion
extrude_len = 5 * np.max(particle_diameters) + insertion_length
insertion_volume = np.pi * (diameter/2)**2 * extrude_len
if "-v" in sys.argv or "--velocity" in sys.argv:
    insertion_velocity = float(sys.argv[sys.argv.index("-v")+1])
    if insertion_velocity < 0:
        insertion_velocity = -insertion_velocity
else:
    insertion_velocity = 0.05
insertion_time = extrude_len/insertion_velocity
if "-n" in sys.argv or "--nparticles" in sys.argv:
    nparticles = int(sys.argv[sys.argv.index("-n")+1])
else:
    # 0.75 is max packing fraction and 0.6 bc we only want to fill 60% of the vessel
    nparticles = int(volume_beaker / average_particle_volume) * 0.75 * 0.6
    # input(
    #    f"nparticles = {nparticles} Volume = {volume_beaker} Average Volume = {average_particle_volume} Press enter to continue")
vol_biggest_particle = 4/3 * np.pi * (np.max(particle_diameters)/2)**3
# this value is very iffy. we can not know how many particles the insertion will actually insert
# thats the reason why we have the beaker in the first place.
# we can only estimate the maximum number of particles that can be inserted
# check if insert_multiplier is in argv
if "-i" in sys.argv or "--insert_multiplier" in sys.argv:
    insert_multiplier = float(sys.argv[sys.argv.index("-i")+1])
else:
    insert_multiplier = 1.0
insertion_max_particles = int(
    insertion_volume / vol_biggest_particle) * 0.35 * insert_multiplier
insertion_rate = insertion_max_particles / insertion_time

# insert at least 1 particle per second
if insertion_rate < 1:
    insertion_rate = 1
print(insertion_rate, insertion_max_particles, insertion_time, nparticles)
# Particle Insertion
insertion_command = f"fix ins all insert/stream seed 32452867 distributiontemplate pdd nparticles {nparticles} particlerate {insertion_rate} overlapcheck yes all_in no vel constant 0.0 0.0 {-insertion_velocity} insertion_face inface extrude_length {extrude_len} \n"
insertion_steps = max(1, np.ceil(nparticles / insertion_max_particles))
time_for_insertion = insertion_time * insertion_steps * 1.1

# sim.execute_command(insertion_command)

# Run simulation up to given time (s)
line = "\n" + "-" * 80 + "\n"

# 0.75 is the position of the insertion face
total_travel_path = extrude_len + height*0.75
total_travel_time = total_travel_path / insertion_velocity
nparticles = 5000
# Inserting Particles
print(line + "Pouring particles" + line)
print(f"Pouring {nparticles} particles with a rate of {insertion_rate} particles per second every {total_travel_time} seconds")
try:
    pos = sim.positions()
    max_z = np.nanmax(pos[:, 2])
except ValueError:
    max_z = 0
while max_z < 0.6*height:
    insertion_command = f"fix ins all insert/stream seed 32452867 distributiontemplate pdd nparticles {nparticles} particlerate {insertion_rate} overlapcheck yes all_in no vel constant 0.0 0.0 {-insertion_velocity} insertion_face inface extrude_length {extrude_len} \n"
    sim.execute_command(insertion_command)
    sim.step_time(total_travel_time*0.95)
    sim.execute_command("unfix ins")
    sim.step_time(total_travel_time*0.05)
    pos = sim.positions()
    max_z = np.nanmax(pos[:, 2])
    print("Max z = ", max_z, "Filled = ", max_z/height)


# Allowing particles to settle
print(line + "Letting remaining particles settle" + line)
reduce_kin_energy(sim)


# First deletion step
radii_before_deletion = sim.radii()

print(line + f"Deleting particles outside 50% region. Round: {1}" + line)

# define region for deletion
# we delete 50% of the volume of the beaker
region_1_command = f"region 1 block {xmin} {xmax} {ymin} {ymax} {height/2} {height} units box\n"
sim.execute_command(region_1_command)
sim.execute_command("delete_atoms region 1")
radii_after_deletion = sim.radii()
number_par_deleted = len(radii_before_deletion) - len(radii_after_deletion)

if number_par_deleted < particle_tolerance:
    raise ValueError(
        "WARNING: No particles deleted in delition step. Can not calculate packing fraction. Increase number of particles or check parameters within this file.")
print(line + "Letting remaining particles uncompact" + line)
reduce_kin_energy(sim)

# Delete particles until none are deleted anymore
i = 1

while number_par_deleted > particle_tolerance:

    radii_before_deletion = sim.radii()

    print(
        line + f"Deleting particles outside 50 ml region. Round: {i+1}" + line)
    sim.execute_command("delete_atoms region 1")
    print(line + "Letting remaining particles settle" + line)

    radii_after_deletion = sim.radii()
    number_par_deleted = len(radii_before_deletion) - len(radii_after_deletion)

    if number_par_deleted < particle_tolerance:
        # if no particles are deleted anymore, we can
        # break the loop and continue with the simulation
        break
    reduce_kin_energy(sim)

    radii_before_deletion = []
    radii_after_deletion = []

    i = 1 + i

# Extract particle properties as NumPy arrays
time = sim.time()
radii = sim.radii()
positions = sim.positions()
velocities = sim.velocities()

print("\n\n" + line)
print(f"Simulation time: {time} s\n")
print(f"Number of particles: {positions.shape}")
print(f"Number of NaN particles: {np.isnan(positions).any(axis = 1).sum()}")
print(f"Number of deletion steps to remove particles above 50 ml: {i}")
print(line + "\n\n")

volume_filled = np.pi * (diameter/2)**2 * height * 0.5
particle_volume = np.sum((4/3) * np.pi * radii**3)

print("Particle Volume fraction: ", particle_volume/volume_filled)
print("Particle Number density: ", len(radii)/volume_filled, " \#/m^3")
print("Particle Bulk density: ", particle_volume *
      density/volume_filled, " kg/m^3")

# Save results as efficient binary NPY-formatted files
np.save(f"{results_dir}/radii.npy", radii)
np.save(f"{results_dir}/positions.npy", positions)
np.save(f"{results_dir}/velocities.npy", velocities)
