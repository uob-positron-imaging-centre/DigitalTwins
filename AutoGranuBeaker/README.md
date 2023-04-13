# Automatic GranuBeaker

The Automatic GranuBeaker is a Discrete Element Method (DEM) simulation   that helps users generate a beaker using gmesh and analyze particle packing based on an input particle size distributions file. The simulation uses the LIGGGHTS DEM engine to perform simulations and provide valuable metrics such as packing density, particle volume fraction, and more.

## Features

- Automatically generate a beaker based on input particle size distribution
- Beaker size is constant factor times bigger than the largest particle radius
- Run simulations using LIGGGHTS
- Calculate packing density, particle volume fraction, and more
- Generate a report containing the results of the simulation

## Requirements

Please make sure you have LIGGGHTS installed on your system. You can find the installation instructions for LIGGGHTS [here](https://www.youtube.com/watch?v=ru3119ozC6M&t=1s&ab_channel=EngineerDo). You also need to install gmesh. You can find the installation instructions for gmesh [here](https://gmsh.info/doc/texinfo/gmsh.html#Installation).
You also need CoExiSt, see [here](https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST) for installation instructions.


## Usage
To use Automatic GranuBeaker, follow these steps:

Prepare a CSV file containing your particle size distribution. The CSV file should have two columns: particle_radius and frequency. (not cumulative frequency)

Run the following command:
```bash
python3 automatic_beaker.py <path_to_csv_file>
```
You have also following options:
- -s, --show: Show the beaker and particles in a 3D plot (gmsh)
- --default or -d, the default 50 ml beaker will be used and no new beaker will be generated. If this flag is not specified, a new beaker will be generated.
- --unit or -u, the unit of the csv file can be specified. Valid units are mm, cm, m and more. Default is m. (e.g. -u mm)
- -o the script automatically generates a psd-file which is readable by ligggths. If -o is specified, the psd-file will be overwritten.
- -n or --nparticles, the number of particles can be specified. If this flag is not specified, the number of particles will be estimated based on the volume of the beaker.

## Contributing to Automatic GranuBeaker
We welcome contributions to Automatic GranuBeaker!

## License

The digital twins hosted in this repository are licensed under [GPL v3.0](https://choosealicense.com/licenses/gpl-3.0/). In non-lawyer terms, the key points of this license are:
- You can view, use, copy and modify this code **_freely_**.
- Your modifications must _also_ be licensed with GPL v3.0 or later.
- If you share your modifications with someone, you have to include the source code as well.

Essentially do whatever you want with the code, but don't try selling it saying it's yours :). This is a community-driven collection building upon many other open-source projects (GCC, LIGGGHTS, even Python itself!) without which this project simply would not have been possible. GPL v3.0 is indeed a very strong *copyleft* license; it was deliberately chosen to maintain the openness and transparency of great software and progress, and respect the researchers pushing granular materials forward. Frankly, open collaboration is way more efficient than closed, for-profit competition.