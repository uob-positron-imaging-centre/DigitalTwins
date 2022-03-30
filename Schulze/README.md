# Schulze Digital Twin for LIGGGHTS Simulations

This repository for the Schulze shear cell contains two different versions:
- ***Normal*** Schulze
- ***Linear*** Schulze

***Note:*** The Schulze and the FT4 simulations share the same python file.
## Normal Schulze
Containing geometry and simulation files for a full-size ShearCell simulation.
This Simulation is with a total fill volume of ***Insert size here*** liter a very intensive simulation. Therefore a smaller, more efficient simulation is provided

## Linear Schulze
The linear schulze contains simulation files and geometry files for a small, linearized version of the Schulze ShearCell. This simulation provides realistic results with only 1/10th of normal schulzes simulation volume and is therefore more efficient.


## Usage

To run ***your*** material in the schulze, simply edit the simulation file (_.sim_) and insert your material properties at the usual positions. (Prior LIGGGHTS knowledge is _strongly_ recommended ). Predefined names should not be changed. The python script uses those variables. Therefore changing: "_N_"  to "_Particle\_number_"   would result in an error.

Then run the python script.

This script is running a pre-defined series of commands which imitate the movements and procedures of the schulze shear cell.


## Todo
- Remeshing of old meshes: Optimization
- Automatic filling to the appropriate level
- Output Section
- Optimize the Functionality: Linear vel change
- Optimize filling procedure
- Parallelize simulation