#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : class.py
# License: GNU v3.0
# Author : Dominik Werner
# Date   : 05.07.2021


import sys
import logging
import numpy as np
from liggghts import liggghts


class Simulation:
    '''Class managing a LIGGGHTS simulation.
    This is made for a diverse set of simulations of common Powder Characterization Tools

    Instantiate it with a path to the `ft4_rheometer.sim` LIGGGHTS script.

    Attributes
    ----------
    filename: str
        name of the Liggghts simulation file. This file defines the system and
        variables like names of walls defined in here are crucial for this program
        make sure to provide the correct names.
        Commands which must not be defined in the LIGGGHTS script:
            Dump command
            Insertion command
    output: str
        output path for LIGGGHTS simulation files
    simulation: LIGGGHTS-instance
        The connection between python and the running LIGGGHTS simulation
    moving_meshes: dict
        a dict containing a lot of information of moving meshes inside the simulation.
        Data saved for each mesh:
            "velocity"
            "rotation"
            "rotation_direction"
            "pid"
            "pid_mode"
            "pid_force"
            "pid_force_axis"
            "pid_max_speed"
            "pid_torque"
            "pid_torque_axis"
            "pid_max_rotation"
            "force_k1"
            "torque_k1"
            "pid_coupled_to"
            "pid_coupled_with"

    dt: float
        timestep of the simulation.
        This is extracted from the LIGGGHTS instance
        the variable "timestep" needs to be defined in the LIGGGHTS script
    n: int
        number of particles in the system
        This is extracted from the LIGGGHTS instance
        the variable "N" needs to be defined in the LIGGGHTS script
    filled: bool
        tells thy script if the system already has been filled or not
    pid_set: bool
        Tells the script if the PID - controller is turned on or not
    pid_number: int
        Number of current PID controllers in the system
    pid_name: STR

    pid_dt
    pid_coupling
    dict
    region

    Methods
    -------


    '''

    def __init__(self, filename, output="."):
        self.filename = filename
        self.output = output

        self.simulation = liggghts()
        self.simulation.file(self.filename)
        logging.basicConfig(filename="pylog.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("Running Urban Planning")

        self.logger = logging.getLogger('Simulation Class')
        # Save particle and blade data
        self.cmd((
            f"dump dmp all custom/vtk 4000 {self.output}/particles_*.vtk "
            "id type type x y z ix iy iz vx vy vz fx fy fz "
            "omegax omegay omegaz radius"
        ))

        # Define the names of the stl's in the liggghts file
        # in order to move them or get the force
        self.moving_meshes = {}
        """
        self.moving_meshes = {
            "blade": {
                "velocity": [0.0, 0.0, 0.0],
                "rotation": 0.0,
                "rotation_direction": 1,
                "pid": False,
                "pid_mode": None,
                "pid_force": None,
                "pid_force_axis": None,
                "pid_max_speed": None,
                "pid_torque": None,
                "pid_torque_axis": None,
                "pid_max_rotation": None,
                "force_k1": None,
                "torque_k1": None,
                "pid_coupled_to": None,
                "pid_coupled_with": None,
            }
        }
        """
        # some declerations (As declared in the file)
        # extract the timestep
        self.dt = self.simulation.extract_variable("timestep", "", 0)

        # extract the number of particles
        self.N = self.simulation.extract_variable("N", "", 0)
        self.filled = False

        # dictionary to keep tract of some things
        # i.e is a mesh already prepared for extraction
        self.dict = {}

        # write the time in a variable
        self.cmd("variable t equal step*dt")

        # number of files written:
        self.n = 0

        # flag to check if pid is set or not
        self.pid_set = False
        self.pid_number = 0
        self.pid_dt = 100
        self.pid_coupling = []
        self.dict = {"vel": [0, 0, 0], "rot": [0, 1]}
        self.region = 0     # number of regions

    def prepare_extraction(
        self,
        name,
        save=True,
        extra_name="data",
        time_to_print=100,
    ):
        """Initiates the output of force torque and position data of a mesh.
        This is done by defining a variables in the LIGGGHTS \
        script and then using the fix print command.

        Parameters
        ----------

        name: str
            name of the mesh to be prepared for extraction
        save: bool
            if True, the data will be saved in a file
        extra_name: str
            an extra extension for the file name
        time_to_print: int
            time in steps after which the data will be printed

        """
        cmds = [
            f"variable fX_{name} equal f_{name}[1]",
            f"variable fY_{name} equal f_{name}[2]",
            f"variable fZ_{name} equal f_{name}[3]",
            f"variable tqX_{name} equal f_{name}[4]",
            f"variable tqY_{name} equal f_{name}[5]",
            f"variable tqZ_{name} equal f_{name}[6]",
            f"variable posZ_{name} equal f_{name}[9]",
        ]

        if save:
            temp_command = \
                f"fix csvfile{self.n} all print {time_to_print} " +\
                "\"$t " + \
                "${" + f"fX_{name}" + "} " +\
                "${" + f"fY_{name}" + "} " +\
                "${" + f"fZ_{name}" + "} " +\
                "${" + f"tqX_{name}" + "} " +\
                "${" + f"tqY_{name}" + "} " +\
                "${" + f"tqZ_{name}" + "} " +\
                "${" + f"posZ_{name}" + "}\" " +\
                f"screen no file {self.output}/{name}_{extra_name}.csv"

            cmds.append(temp_command)
            self.n += 1

        for cmd in cmds:
            self.cmd(cmd)

        self.dict[name] = True

    def move_manager(self, name, rotation=None, move=None):
        """ Manages the movement of a mesh. Necessary due to the limitations and complexity of LIGGGHTS.
        This function keeps track of all movements of all meshes and automatically updates the simulation of needed.

        Liggghts is
        Parameters
        ----------

        name: str
            name of the mesh to be moved
        rotation: float
            rotation of the mesh in degrees
        move: list
            list of floats [x, y, z]
            x, y, z are the distances to move the mesh in each direction


        """
        if name not in self.moving_meshes:
            self.moving_meshes[name] = {
                "velocity": [0.0, 0.0, 0.0],
                "rotation": 0.0,
                "rotation_direction": 1,
                "pid": False,
                "pid_mode": None,
                "pid_force": None,
                "pid_force_axis": None,
                "pid_max_speed": None,
                "pid_torque": None,
                "pid_torque_axis": None,
                "pid_max_rotation": None,
                "force_k1": None,
                "torque_k1": None,
                "pid_coupled_to": None,
                "pid_coupled_with": None,
            }

        # First unset all move commands.
        # must happen in specific order.
        for dict_name in self.moving_meshes:
            speed_x, speed_y, speed_z = \
                self.moving_meshes[dict_name]["velocity"]

            rpm = self.moving_meshes[dict_name]["rotation"]
            direction = self.moving_meshes[dict_name]["rotation_direction"]

            # there is no move command set if all velocitys are 0 or the
            # rotation is 0
            print(rpm)
            if rpm != 0:
                self.cmd(f"unfix rotate_{dict_name}")
            if speed_x != 0 and speed_y != 0 and speed_z != 0:
                self.cmd(f"unfix move_{dict_name}")

        # Now set all move commands again
        # this must happen in reversed order than unset.
        for dict_name in reversed(self.moving_meshes):
            # get new values
            if name == dict_name and rotation is not None:
                rpm, direction = rotation
                self.moving_meshes[dict_name]["rotation"] = rpm
                self.moving_meshes[dict_name]["rotation_direction"] = direction
            else:
                rpm = self.moving_meshes[dict_name]["rotation"]
                direction = self.moving_meshes[dict_name]["rotation_direction"]
            if name == dict_name and move is not None:
                speed_x, speed_y, speed_z = move
                self.moving_meshes[dict_name]["velocity"] = \
                    [speed_x, speed_y, speed_z]
            else:
                speed_x, speed_y, speed_z = \
                    self.moving_meshes[dict_name]["velocity"]

            # set the values in reverse direction
            self.set_speed(dict_name, speed_x, speed_y, speed_z)
            self.set_rotation(dict_name, rpm, direction)

    def save_state(self, new_filename):
        """ Save the current state of the simulation. Using the LIGGGHTS \
        write_restart command.

        """
        self.cmd(f"write_restart {self.output}/{new_filename}")

    def load_state(self, filename):
        """ load a state of the simulation. Using the LIGGGHTS \
        read_restart command.
        """
        with open(f"{self.filename}", "r") as f:
            sim_file = f.readlines()

        for line_id, line in enumerate(sim_file):
            if line.startswith("create_box"):
                sim_file[line_id] = f"read_restart {self.output}/{filename}"
                break

        with open(f"{self.output}/temp.sim", "w") as f:
            f.writelines(sim_file)

        del self.simulation
        self.simulation = liggghts()
        self.simulation.file(f"{self.output}/temp.sim")
        self.cmd("variable t equal step*dt")

    def get_torque(self, name):
        """ Returns the torque of a mesh.
        Parameters
        ----------
        name: str
            name of the mesh

        """
        if name in self.dict:
            if self.dict[name] is True:
                torque = [
                    self.simulation.extract_variable(f"tqX_{name}", "", 0),
                    self.simulation.extract_variable(f"tqY_{name}", "", 0),
                    self.simulation.extract_variable(f"tqZ_{name}", "", 0),
                ]
        else:
            print("Unprepared mesh file!")
            torque = None

        return torque

    def get_force(self, name):
        """ Returns the force of a mesh.
        Parameters
        ----------
        name: str
            name of the mesh
        """
        if name in self.dict:
            if self.dict[name] is True:
                force = [
                    self.simulation.extract_variable(f"fX_{name}", "", 0),
                    self.simulation.extract_variable(f"fY_{name}", "", 0),
                    self.simulation.extract_variable(f"fZ_{name}", "", 0),
                ]
        else:
            raise ValueError("Unprepared mesh file! can not read force data")

        return force

    def move_distance(self, name, speed, distance):
        """ Moves a mesh a given distance.
        NOTE: THIS RUNS THE SIMULATION FOR SOME TIMESTEPS

        Parameters
        ----------
        name: str
            name of the mesh
        speed: list
            list of floats [vx, vy, vz]
            vx, vy, vz are the speed to move the mesh in each direction
        distance: list
            list of floats [x, y, z]
            x, y, z are the distances to move the mesh in each direction
        """
        # move the rod to a specific height of the box
        self.move_manager(name, move=[speed[0], speed[1], speed[2]])
        timesteps = round(np.nanmin(
            np.asarray(distance) / np.asarray(speed)
        ) / self.dt)

        self.run(timesteps)
        self.move_manager(name, move=[0, 0, 0])

    def set_speed(self, name, speed_x=0, speed_y=0, speed_z=0):

        if speed_x == 0 and speed_y == 0 and speed_z == 0:
            return 0
        else:
            # set the rod speed
            x = float(speed_x)
            y = float(speed_y)
            z = float(speed_z)
            cmd = (
                f"fix move_{name} all move/mesh mesh {name} linear {x} {y} {z}"
            )

            self.cmd(cmd)

    def push_with_force(self, force, dump=False, time=100000):
        """ Include a Servo Mesh in LIGGGHTS. This mesh presses particles with a given \
        force.
        Parameters
        ----------
        force: list
            list of floats [fx, fy, fz]
            fx, fy, fz are the force to push the mesh with in each direction
        dump: bool
            if True, the servo wall will be saved in a file
        time: int
            time in timesteps to run the simulation and press the particles down.

        """

        cmd = (
            "fix servo all mesh/surface/stress/servo file mesh/plate.stl "
            "type 1 com 0. 0. 0.1 ctrlPV force axis 0. 0. -1. target_val "
            f"{force} vel_max 0.1 kp 5."
        )
        self.cmd(cmd)
        self.cmd("unfix granwalls")

        cmd = (
            "fix granwalls all wall/gran model hertz tangential history "
            "cohesion sjkr rolling_friction cdt  mesh n_meshes 3 "
            "meshes cad servo blade"
        )
        self.cmd(cmd)

        if dump:
            cmd2 = (
                f"dump dmpplate all mesh/vtk 2000 {self.output}/plate_*.vtk "
                "servo stress wear"
            )
            self.cmd(cmd2)

        # Press cycle:
        self.run(time - 30000)

        # Now we need a release cycle which is 20000 timesteps long
        self.cmd("fix_modify servo ctrlParam 0.2 0.0 0.0")
        f = force / 10
        self.cmd(f"fix_modify servo target_val {f} ")
        self.run(20000)

        f2 = force / 100000
        self.cmd(f"fix_modify servo target_val {f2} ")
        self.run(10000)
        if dump:
            self.cmd("undump dmpplate")

        self.cmd("unfix granwalls")
        self.cmd("unfix servo")
        cmd = (
            "fix granwalls all wall/gran model hertz tangential history "
            "cohesion sjkr rolling_friction cdt  mesh n_meshes 2 "
            "meshes cad  blade"
        )
        self.cmd(cmd)

    def set_rotation(self, name, rpm, direction=1):
        """Set the rotation of a mesh.
         Parameters
        ----------
        name: str
            name of the mesh
        rpm: float
            rotation speed in rpm
        direction: int
            direction of rotation. 1 for clockwise and -1 for counterclockwise
        """
        if rpm == 0:
            return 0
        else:
            rot = 60 / rpm  # converting rpm to seconds per revolution
            direction = float(np.sign(direction))
            cmd = (
                f"fix rotate_{name} all move/mesh mesh {name} rotate origin "
                f"0. 0. 0. axis  0. 0. {direction} period {rot}"
            )
            self.cmd(cmd)

    def cmd(self, cmd):
        """ Run a command in LIGGGHTS.
        Parameters
        ----------
        cmd: str
            command to run in LIGGGHTS
        """
        self.logger.info(f"Running command: {cmd}")
        self.simulation.command(cmd)

    def run(self, timesteps):
        """ Run the simulation for a given number of timesteps.
        Parameters
        ----------
        timesteps: int
            number of timesteps to run the simulation
        """
        timesteps = int(timesteps)
        t = 0
        if self.pid_set:
            for t in range(0, timesteps, self.pid_dt):
                self.pid_worker()
                self.cmd(f"run {self.pid_dt}")
        timesteps -= t
        self.cmd(f"run {timesteps}")

    def fill(self, max_runs=150):
        """ Fill the simulation with particles.
        The number of particles filled into the system is equal to the variable \"N\"\
        in the LIGGGHTS input file.
        Parameters
        ----------
        max_runs: int
            number of runs to fill the simulation with particles.
            A run consists of 1000 timesteps.
        """
        runs = 0
        if self.filled:
            print("already filled!!")
        else:
            self.filled = True
            self.simulation.command((
                "fix ins all insert/stream seed 67867967 distributiontemplate "
                "pdd nparticles ${N} particlerate ${rate} overlapcheck yes "
                "all_in no vel constant 0.0 0.0 -0.5 insertion_face inface "
                "extrude_length 0.05 "
            ))

            while True:
                runs += 1
                N = self.simulation.extract_global("natoms", 0)

                if N != self.N:
                    self.simulation.command("run 1000")
                else:
                    break
                if runs >= max_runs:
                    print("Max number of runs!")
                    break

    def pid_init(
        self,
        name,
        force=[0.0, 0.0, 0.0],
        max_velocity=0.0,
        torque=None,
        torque_axis=None,
        max_rotation=None,
        axis=2,
        force_k=0.1,
        torque_k=0.1,
        coupling=None
    ):
        # init mesh if never used
        if name not in self.moving_meshes:
            self.moving_meshes[name] = {
                "velocity": [0.0, 0.0, 0.0],
                "rotation": 0.0,
                "rotation_direction": 1,
                "pid": False,
                "pid_force": None,
                "pid_max_speed": None,
                "pid_torque": None,
                "pid_torque_axis": None,
                "pid_max_rotation": None,
                "force_k1": None,
                "torque_k1": None,
                "pid_coupled_to": None,
                "pid_coupled_with": None,
            }

        self.moving_meshes[name]["pid"] = True
        self.moving_meshes[name]["pid_force"] = force
        self.moving_meshes[name]["pid_max_speed"] = max_velocity
        self.moving_meshes[name]["pid_torque"] = torque
        self.moving_meshes[name]["pid_torque_axis"] = torque_axis
        self.moving_meshes[name]["pid_max_rotation"] = max_rotation
        self.moving_meshes[name]["pid_coupled_to"] = coupling

        if coupling is not None:
            # set it to false as all the calculations are beeing done with
            # the "master" mesh
            self.moving_meshes[name]["pid"] = False
            self.moving_meshes[coupling]["pid_coupled_with"] = name

        self.moving_meshes[name]["force_k1"] = force_k
        self.moving_meshes[name]["torqu_k1"] = torque_k
        self.pid_set = True
        self.pid_number += 0

    def pid_unset(self, name):
        self.moving_meshes[name]["pid"] = False
        self.moving_meshes[name]["pid_force"] = [0.0, 0.0, 0.0]
        self.moving_meshes[name]["pid_max_speed"] = None
        self.moving_meshes[name]["pid_torque"] = None
        self.moving_meshes[name]["pid_torque_axis"] = None
        self.moving_meshes[name]["pid_max_rotation"] = None
        self.moving_meshes[name]["pid_coupled_to"] = None
        self.moving_meshes[name]["force_k1"] = None
        self.moving_meshes[name]["torqu_k1"] = None
        self.move_manager(name, move=[0, 0, 0])
        self.move_manager(name, rotation=[0, 1])
        self.pid_number -= 1
        if self.pid_number == 0:
            self.pid_set = False
        pass

    def pid_worker(self):
        for name in self.moving_meshes:

            if not self.moving_meshes[name]["pid"]:
                continue

            if self.moving_meshes[name]["pid_force"] is not [0.0, 0.0, 0.0]:
                target = self.moving_meshes[name]["pid_force"]
                force = np.asarray(self.get_force(name))
                if self.moving_meshes[name]["pid_coupled_with"] is not None:
                    # if a coupling is active we need to consider the force
                    # from the coupled file too!
                    force += np.asarray(
                        self.get_force(
                            self.moving_meshes[name]["pid_coupled_with"]
                        )
                    )

                vel = self.moving_meshes[name]["velocity"]
                for axis in range(3):
                    if target[axis] == 0.0 or target[axis] is None:
                        continue

                    print(force, target)
                    delta_force = force[axis] - target[axis]
                    new_speed = self.moving_meshes[name]["force_k1"] * \
                        delta_force
                    print(new_speed, self.moving_meshes[name]["pid_max_speed"])

                    pid_max_speed = self.moving_meshes[name]["pid_max_speed"]

                    if abs(new_speed) >= pid_max_speed:
                        new_speed = abs(new_speed) / new_speed * pid_max_speed

                    vel[axis] = new_speed

                self.move_manager(name, move=[vel[0], vel[1], vel[2]])

            if self.moving_meshes[name]["pid_torque"] is not None:
                target = self.moving_meshes[name]["pid_torque"]
                print(self.moving_meshes)

                torque = self.get_torque(
                    name
                )[self.moving_meshes[name]["pid_torque_axis"]]

                if self.moving_meshes[name]["pid_coupled_with"] is not None:
                    # If a coupling is active we need to consider the torque
                    # from the coupled file too!
                    torque += self.get_torque(
                        self.moving_meshes[name]["pid_coupled_with"]
                    )[self.moving_meshes[name]["pid_torque_axis"]]

                delta_force = torque - target
                new_speed = self.moving_meshes[name]["torqu_k1"] * delta_force

                pid_max_rotation = self.moving_meshes[name]["pid_max_rotation"]
                if abs(new_speed) >= pid_max_rotation:
                    new_speed = abs(new_speed) / new_speed * pid_max_speed

                rpm = abs(new_speed)
                direction = (abs(new_speed) / new_speed)
                # input(vel)

                self.move_manager(name, rotation=[rpm, direction])

        for name in self.moving_meshes:
            if self.moving_meshes[name]["pid_coupled_to"] is not None:
                # if coupling is active use values from the coupled mesh
                target_vel = self.moving_meshes[
                    self.moving_meshes[name]["pid_coupled_to"]
                ]["velocity"]

                target_rotation = self.moving_meshes[
                    self.moving_meshes[name]["pid_coupled_to"]
                ]["rotation"]

                target_rotation_direction = self.moving_meshes[
                    self.moving_meshes[name]["pid_coupled_to"]
                ]["rotation_direction"]

                self.move_manager(name, move=target_vel, rotation=[
                    target_rotation, target_rotation_direction
                ])

    def delete_region(
        self,
        xlo="INF",
        xhi="INF",
        ylo="INF",
        yhi="INF",
        zlo="INF",
        zhi="INF",
    ):
        """ Delete particles in a region.

        Parameters:
        -----------
        xlo: float
            Lower x bound.
        xhi: float
            Upper x bound.
        ylo: float
            Lower y bound.
        yhi: float
            Upper y bound.
        zlo: float
            Lower z bound.
        zhi: float
            Upper z bound.
        """
        self.region += 1
        cmd = (
            f"region del_region{self.region} block "
            f"{xlo} {xhi} {ylo} {yhi} {zlo} {zhi}"
        )

        self.cmd(cmd)
        cmd = f"delete_atoms region del_region{self.region} "

        self.cmd(cmd)

    def test(self):
        """ Simple Test function
        """
        self.fill()
        self.prepare_extraction("blade", save=True)
        self.run(1000)
        print(self.get_force("blade"))
        print(self.get_torque("blade"))
        # self.set_rotation("blade", 60)
        self.run(100)

        # Move 10 cm with 10 cm / s
        self.move_distance("blade", [0, 0, -0.1], [0, 0., -0.1])
        self.run(10000)
        self.set_rotation("blade", 60)
        self.run(100000)
        self.set_rotation("blade", 0)
        self.run(10000)
        self.set_rotation("blade", 60, -1)
        self.run(100000)
        self.set_rotation("blade", 60)
        self.run(10000)

        # Move 10 cm with 10 cm / s
        self.move_distance("blade", [0, 0, 0.1], [0, 0., 0.1])

    def test2(self):
        """ Simple Test function
        """
        self.fill()
        self.prepare_extraction("blade", save=True)
        self.run(100000)
        self.push_with_force(1000, dump=True, time=100000)
        self.run(100000)
        self.move_distance("blade", [0, 0, -0.1], [0, 0., -0.1])

    def test3(self):
        """ Simple Test function
        """
        self.fill()
        self.run(100)

        self.prepare_extraction("blade", save=True)
        self.prepare_extraction("shear_head", save=True)
        self.prepare_extraction("shear_blades", save=True)
        self.pid_init(
            "blade",
            torque=0.01,
            torque_axis=2,
            max_rotation=100,
            torque_k=100,
        )

        self.run(500)
        self.set_rotation("blade", 30, direction=-1)

        self.run(200)
        self.set_rotation("blade", 30, direction=1)

        self.run(200)
        self.set_rotation("blade", 0, direction=1)

        self.pid_unset("blade")
        self.pid_init(
            "blade",
            force=[0, 0, 100],
            max_velocity=0.1,
            force_k=0.1,
        )

        self.run(2000)
        self.pid_unset("blade")
        self.pid_init(
            "shear_head",
            force=[0, 0, 1000],
            max_velocity=0.1,
            force_k=0.1,
        )
        self.pid_init(
            "shear_blades",
            force=[0, 0, 1000],
            max_velocity=0.1,
            force_k=0.1,
            coupling="shear_head",
        )

        self.run(2000)
        self.pid_unset("blade")
        self.pid_init(
            "blade",
            force=[0, 0, 1002],
            max_velocity=0.1,
            force_k=0.1,
        )
        self.run(2000)

    def rheometer_run(
        self,
        rpm_down,
        rpm_up,
        lin_down,
        lin_up,
        upwards=True,
    ):
        """ FT4 - Rheometer run.
        Run the rheometer with a given RPM and linear speed up and down.
        Force and Torque data is extracted and saved to a file.

        Parameters
        ----------
        rpm_down : float
            RPM of the rheometer downwards.
        rpm_up : float
            RPM of the rheometer upwards.
        lin_down : float
            Linear speed of the rheometer downwards.
        lin_up : float
            Linear speed of the rheometer upwards.
        upwards : bool
            If True, the rheometer is also run upwards.
        """
        # save mesh
        self.cmd((
            f"dump dmpstl all mesh/vtk 4000 {self.output}/blade_*.vtk "
            "blade stress wear"
        ))

        self.fill()
        self.run(20000)
        self.delete_region(zlo=0.0796)

        # Move the blade to the top of the particle bed
        self.move_distance("blade", [0.0, 0.0, -0.1], [0.0, 0.0, -0.02])

        # Prepare extraction of height, torque, forces, etc.
        self.prepare_extraction("blade", save=True)
        self.prepare_extraction("cad", save=True)

        # Start moving the blade downwards
        self.move_manager("blade", rotation=[rpm_down, 1])
        self.move_distance("blade", [0.0, 0.0, lin_down], [0.0, 0.0, -0.0755])

        # Stop movement
        if upwards:
            self.move_manager("blade", rotation=[0, 1])
            self.move_manager("blade", move=[0, 0, 0])

            # Start moving the blade upwards
            self.run(20000)
            self.move_manager("blade", rotation=[rpm_up, -1])
            self.move_distance("blade", [0.0, 0.0, lin_up], [0.0, 0.0, 0.1])
            self.move_manager("blade", rotation=[0, 1])

    def ft4_shear_cell_run(self):
        """ FT4 - Shear cell run.
        A pre-defined shear cell run pushing with 2, 1.75, 1.5, 1.25, 1 kPa.
        This script fills the simulation and saves it as a presheared file to make \
        most use out of the simulations!
        """
        self.cmd((
            f"dump dmpstl all mesh/vtk 4000 {self.output}/blade_*.vtk "
            "blade stress wear"
        ))
        self.cmd((
            f"dump dmpstl2 all mesh/vtk 4000 {self.output}/shear_head_*.vtk "
            "shear_head shear_blades stress wear"
        ))

        area = np.pi / 4 * 48e-3**2

        # Settings:
        rpm = 5
        rotations = 0.125       # number of rotations to measure shear

        # Number of timesteps needed to rotate
        time_to_rotate = int(rotations / (rpm / 60) / self.dt)
        print(time_to_rotate)

        # Set false if you already have a precompressed file!
        if True:
            self.fill()
            self.run(10000)
            self.delete_region(zlo=0.085)
            self.push_with_force(1000)
            self.delete_region(zlo=0.07)
            self.save_state("precompressed.simsave")

        self.load_state("precompressed.simsave")
        self.prepare_extraction("blade", save=True)
        self.prepare_extraction("shear_head", save=True)
        self.prepare_extraction("shear_blades", save=True)

        # Push shear head into particles with 3kpa
        self.pid_init(
            "shear_head",
            force=[0, 0, 3000 * area],    # convert pressure to force
            max_velocity=0.1,             # max speed is 0.1 m/s
            force_k=0.02,                 # control constant
        )

        # Couple the blades to the head
        self.pid_init("shear_blades", coupling="shear_head")

        # Run a while (check later until the velocity = 0
        self.run(100000)

        # when particle contact shear
        self.move_manager("shear_head", rotation=[rpm, 1])
        self.move_manager("shear_blades", rotation=[rpm, 1])
        self.run(time_to_rotate)

        # shear back to initial position:
        self.move_manager("shear_head", rotation=[rpm, -1])
        self.move_manager("shear_blades", rotation=[rpm, -1])
        self.run(time_to_rotate)

        # stop rotation
        self.move_manager("shear_head", rotation=[0, -1])
        self.move_manager("shear_blades", rotation=[0, -1])
        self.run(10000)
        self.save_state("pre_sheared.simsave")

        for normal_pressure in [2, 1.75, 1.5, 1.25, 1]:

            self.load_state("pre_sheared.simsave")

            # After load state we need to reinit the meshes
            self.prepare_extraction(
                "blade",
                save=True,
                extra_name=f"n_stress_{normal_pressure}_kpa",
            )

            self.prepare_extraction(
                "shear_head",
                save=True,
                extra_name=f"n_stress_{normal_pressure}_kpa",
            )

            self.prepare_extraction(
                "shear_blades",
                save=True,
                extra_name=f"n_stress_{normal_pressure}_kpa",
            )

            self.run(1000)

            # Convert pressure to force
            self.pid_init(
                "shear_head",
                force=[0, 0, normal_pressure * 1000 * area],
                max_velocity=0.1,     # max speed is 0.1 m/s
                force_k=0.02,         # control constant
            )

            # particles should be already in contact
            # shear now
            self.run(10000)
            self.move_manager("shear_head", rotation=[rpm, 1])
            self.move_manager("shear_blades", rotation=[rpm, 1])
            self.run(time_to_rotate)

            # shear back to initial position:
            self.move_manager("shear_head", rotation=[rpm, -1])
            self.move_manager("shear_blades", rotation=[rpm, -1])
            self.run(time_to_rotate)

            # stop rotation
            self.move_manager("shear_head", rotation=[0, -1])
            self.move_manager("shear_blades", rotation=[0, -1])
            self.run(10000)
            # finish, next run

    def schulze_box_vel(self, vel):
        """ Set the Linear velocity of the Schulze box.

        Parameters
        ----------
        vel : float
            Linear velocity of the Schulze.
        """
        self.cmd("unfix granwalls")
        self.cmd((
            "fix cad all mesh/surface/stress file  mesh/linear_box.stl  "
            "heal auto_remove_duplicates type 2 scale 0.001 surface_vel "
            f"{vel} 0. 0. wear finnie "
        ))

        self.cmd((
            "fix granwalls all wall/gran model hertz tangential history "
            "cohesion sjkr rolling_friction cdt  mesh n_meshes 3 meshes cad "
            "shear_head shear_blades"
        ))

    def schulze_linear(self):
        """ Schulze linear script.
        This script runs shear tests with the linear Schulze.
        Pressures applied: 2020, 1420, 920, 420 Pa
        """
        area = 0.048 * 0.046
        velocity = 0.1
        time_to_rotate = 150000

        self.fill()
        self.prepare_extraction("shear_head", save=True)
        self.prepare_extraction("shear_blades", save=True)
        self.pid_init(
            "shear_head",
            force=[0, 0, 3000 * area],    # convert pressure to force
            max_velocity=0.1,             # max speed is 0.1 m/s
            force_k=0.02                  # control constant
        )

        # couple the blades to the head
        self.pid_init("shear_blades", coupling="shear_head")

        # self.run(40000)
        self.schulze_box_vel(velocity)
        self.run(30000)
        self.schulze_box_vel(0.0)
        self.save_state("pre_sheared.simsave")

        for normal_pressure in [2020, 1420, 920, 420]:
            self.load_state("pre_sheared.simsave")

            # after load state we need to reinit the meshes
            self.prepare_extraction(
                "shear_head",
                save=True,
                extra_name=f"n_stress_{normal_pressure}_kpa",
            )

            self.prepare_extraction(
                "shear_blades",
                save=True,
                extra_name=f"n_stress_{normal_pressure}_kpa",
            )

            self.run(5000)

            # Convert pressure to force
            self.pid_init(
                "shear_head",
                force=[0, 0, normal_pressure * area],
                max_velocity=0.1,    # max speed is 0.1 m/s
                force_k=0.02,         # control constant
            )

            self.pid_init("shear_blades", coupling="shear_head")

            # particles should be already in contact
            # shear now
            self.run(5000)
            self.schulze_box_vel(velocity)
            self.run(time_to_rotate)

    def schulze_normal(self):
        r1 = 0.1  # 10 cm radius
        r2 = 0.05
        area = np.pi * (r1**2 - r2**2)
        velocity = 0.1
        time_to_rotate = 50000
        rpm = 5
        self.fill()
        self.prepare_extraction("shear_head", save=True)
        self.prepare_extraction("shear_blades", save=True)
        self.pid_init(
            "shear_head",
            force=[0, 0, 3000 * area],  # convert pressure to force
            max_velocity=0.1,  # max speed is 0.1 m/s
            force_k=0.015  # control constant
        )
        # couple the blades to the head
        self.pid_init("shear_blades", coupling="shear_head")
        self.run(20000)
        self.move_manager("cad", rotation=[rpm, 1])
        self.run(10000)
        self.move_manager("cad", rotation=[rpm, -1])
        self.run(10000)
        self.move_manager("cad", rotation=[0, 1])
        self.save_state("pre_sheared.simsave")

        for normal_pressure in [2020, 1420, 920]:
            self.load_state("pre_sheared.simsave")

            # after load state we need to reinit the meshes
            self.prepare_extraction(
                "shear_head", save=True, extra_name=f"n_stress_{normal_pressure}_kpa")
            self.prepare_extraction(
                "shear_blades", save=True, extra_name=f"n_stress_{normal_pressure}_kpa")
            self.run(1000)
            self.pid_init(
                "shear_head",
                # convert pressure to force
                force=[0, 0, normal_pressure * area],
                max_velocity=0.1,  # max speed is 0.1 m/s
                force_k=0.0015  # control constant
            )
            self.pid_init("shear_blades", coupling="shear_head")
            self.run(1000)
            # particles should be already in contact
            # shear now
            self.move_manager("cad", rotation=[rpm, 1])
            self.run(time_to_rotate)
            self.move_manager("cad", rotation=[0, 1])

    def __del__(self):
        """ Close the simulation.
        """
        self.simulation.close()
        print("Goodbye! :-) ")


if __name__ == "__main__":
    # Usage:
    # sim = FT4Rheometer("liggghts_script", "output_folder")
    # sim.rheometer_run(rpm_down, rpm_up, lin_down, lin_up)

    # Use command-line arguments if supplied, otherwise use defaults
    script_path = sys.argv[1] if len(sys.argv) >= 2 else "schulze_lin.sim"
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "sim_outputs"

    sim = Simulation(script_path, output_path)
    sim.schulze_linear()
