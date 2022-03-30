# FT4 Digital Twin for LIGGGHTS Simulations

This repository for the FT4 offers two different running modes:
- Shear Cell mode
- Rheometer mode

To switch between modes, see the file [class.py](class.py).
In the end section, choose functions "_ft4\_rheometer\_run_" or "_ft4\_shear\_cell\_run_".
See function definitions for more details.


***Note:*** The Schulze and the FT4 simulations share the same python file.



## Usage

To run ***your*** material in the Ft4, simply edit the simulation file (_.sim_) and insert your material properties at the usual positions. (Prior LIGGGHTS knowledge is _strongly_ recommended ). Predefined names should not be changed. The python script uses those variables. Therefore changing: "_N_"  to "_Particle\_number_"   would result in an error.

Then run the python script.


## Todo
- Remeshing of old meshes: Optimization
- Automatic filling to the appropriate level
- Output Section
- Optimize the Functionality: Linear vel change
- Optimize filling procedure
- Parallelize simulation
