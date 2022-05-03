import os
import shutil


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# Orifice Sizes
orifice_size = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # mm

# Directory to save temporary simulation files to
scriptdir = "simulation_files"
if not os.path.isdir(scriptdir):
    os.mkdir(scriptdir)

# Create directory to save results to
for size in orifice_size:
    results_dir = f'results/Orifice_{size:02d}mm'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

# Directory paths

template_dir = 'Templates'
simulation_dir = 'simulation_files'

# Create individual simulation files.

for i in range(len(orifice_size)):

    with open(f'{template_dir}/simulation_script_template.py') as f:
        sim_input = f.readlines()

    sim_input[22] = f"orifice_size = {orifice_size[i]}\n"

    with open(f'{scriptdir}/simulation_script_{orifice_size[i]}mm.py', "w") as f:
        f.writelines(sim_input)

    # Create the run.sh files for BlueBear.

    with open(f'{template_dir}/run_simulation_template.sh', "r") as f:
        lines = f.readlines()

    lines[37] = f"python3 simulation_script_{orifice_size[i]}mm.py"
    batch_filename = f"run_simulation_{orifice_size[i]}mm.sh"

    with open(f'{simulation_dir}/run_simulation_{orifice_size[i]}mm.sh', "w") as f:
        f.writelines(lines)

    with cd("simulation_files"):
        os.system(f"sbatch {batch_filename}")
