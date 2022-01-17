'''ACCESS Example User Simulation Script

Must define one simulation whose parameters will be optimised this way:

    1. Use a variable called "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       You can set the initial guess here.

    2. The `parameters` creation should be fully self-contained between
       `#### ACCESS PARAMETERS START` and `#### ACCESS PARAMETERS END`
       blocks (i.e. it should not depend on code ran before that).

    3. By the end of the simulation script, define a variable called `error` -
       one number representing this simulation's error value.

Importantly, use `parameters.at[<free parameter name>, "value"]` to get this
simulation's free / optimisable variable values.
'''

# Either run the actual GranuDrum simulation (takes ~40 minutes) or extract
# pre-computed example data and instantly show error value and plots
run_simulation = False      # Run simulation (True) or use example data (False)
save_data = True            # Save particle positions, radii and timestamps
show_plots = True           # Show plots of simulated & experimental GranuDrum


#### ACCESS PARAMETERS START
import attr
from typing import Tuple

import numpy as np
import cv2

import coexist
import konigcell as kc      # For occupancy grid computation

parameters = coexist.create_parameters(
    variables = ["sliding", "rolling", "nparticles"],
    minimums = [0., 0., 15494],
    maximums = [2., 1., 36152],
    values = [0.3197, 0.00248, 29954],
)

access_id = 0               # Unique ID for each ACCESS simulation run
#### ACCESS PARAMETERS END

# Extract current free parameters' values
sliding = parameters.at["sliding", "value"]
rolling = parameters.at["rolling", "value"]
nparticles = parameters.at["nparticles", "value"]

# Create a new LIGGGHTS simulation script with the parameter values above; read
# in the simulation template and change the relevant lines
with open("granudrum_mcc.sim", "r") as f:
    sim_script = f.readlines()

sim_script[0] = f"log simulation_inputs/granudrum_mcc_{access_id:0>6}.log\n"

sim_script[16] = f"variable fricPP equal {sliding}\n"
sim_script[17] = f"variable fricPW equal {sliding}\n"
sim_script[18] = f"variable fricPSW equal {sliding}\n"

sim_script[21] = f"variable fricRollPP equal {rolling}\n"
sim_script[22] = f"variable fricRollPW equal {rolling}\n"
sim_script[23] = f"variable fricRollPSW equal {rolling}\n"

sim_script[9] = f"variable N equal {nparticles}\n"

# Save the simulation template with the changed free parameters
filepath = f"simulation_inputs/granudrum_mcc_{access_id:0>6}.sim"
with open(filepath, "w") as f:
    f.writelines(sim_script)

# Load simulation and run it for two GranuDrum rotations. Use the last quarter
# rotation to compute the error value
rpm = 45
rotations = 2
start_time = (rotations - 0.25) / (rpm / 60)
end_time = rotations / (rpm / 60)


if run_simulation:
    sim = coexist.LiggghtsSimulation(filepath, verbose = True)

    # Record times, radii and positions at 120 FPS
    checkpoints = np.arange(start_time, end_time, 1 / 120)

    times = []
    positions = []
    radii = []

    for t in checkpoints:
        sim.step_to_time(t)

        times.append(sim.time())
        radii.append(sim.radii())
        positions.append(sim.positions())

    times = np.array(times)             # 1D array (Timestep,)
    radii = np.array(radii)             # 2D array (Timestep, Radius)
    positions = np.array(positions)     # 3D array (Timestep, Particle, XYZ)

    if save_data:
        np.save(f"example_positions/time_{access_id}.npy", times)
        np.save(f"example_positions/radii_{access_id}.npy", radii)
        np.save(f"example_positions/positions_{access_id}.npy", positions)
else:
    # Load example simulated data
    times = np.load(f"example_positions/time_{access_id}.npy")
    radii = np.load(f"example_positions/radii_{access_id}.npy")
    positions = np.load(f"example_positions/positions_{access_id}.npy")




@attr.s(auto_attribs = True)
class GranuDrum:
    '''Class encapsulating a GranuTools GranuDrum system dimensions, assumed to
    be centred at (0, 0).
    '''
    xlim: list = [-0.042, +0.042]
    ylim: list = [-0.042, +0.042]
    radius: float = 0.042




def encode_u8(image: np.ndarray) -> np.ndarray:
    '''Convert image from doubles to uint8 - i.e. encode real values to
    the [0-255] range.
    '''

    u8min = np.iinfo(np.uint8).min
    u8max = np.iinfo(np.uint8).max

    img_min = float(image.min())
    img_max = float(image.max())

    img_bw = (image - img_min) / (img_max - img_min) * (u8max - u8min) + u8min
    img_bw = np.array(img_bw, dtype = np.uint8)

    return img_bw


def image_thresh(
    granudrum: GranuDrum,
    image_path: str,
    trim: float = 0.7,
) -> kc.Pixels:
    '''Return the raw and thresholded GranuDrum image in the `konigcell.Pixels`
    format.
    '''

    # Load the image in grayscale and ensure the orientation is:
    #    - x is downwards
    #    - y is rightwards
    image = 255 - cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[::-1].T

    # Pixellise / discretise the GranuDrum circular outline
    xgrid = np.linspace(granudrum.xlim[0], granudrum.xlim[1], image.shape[0])
    ygrid = np.linspace(granudrum.ylim[0], granudrum.ylim[1], image.shape[1])

    # Physical coordinates of all pixels
    xx, yy = np.meshgrid(xgrid, ygrid)

    # Remove the GranuDrum's circular outline
    image[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((11, 11), np.uint8)
    image2 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Global thresholding + binarisation
    _, image2 = cv2.threshold(image2, 30, 255, cv2.THRESH_BINARY)
    image2[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    image = kc.Pixels(image, xlim = granudrum.xlim, ylim = granudrum.ylim)
    image2 = kc.Pixels(image2, xlim = granudrum.xlim, ylim = granudrum.ylim)

    # Return the original and binarised images
    return image, image2


def simulation_thresh(
    granudrum: GranuDrum,
    image_shape: Tuple[int, int],
    times: np.ndarray,
    radii: np.ndarray,
    positions: np.ndarray,
    trim: float = 0.7,
) -> kc.Pixels:
    '''Return the raw and thresholded occupancy grid of the GranuDrum DEM
    simulation in the `konigcell.Pixels` format.
    '''

    # Extract GranuDrum dimensions
    xlim = granudrum.xlim
    ylim = granudrum.ylim

    # Concatenate every particle trajectory, separated by a row of NaNs
    num_timesteps = positions.shape[0]
    num_particles = positions.shape[1]

    positions = np.swapaxes(positions, 0, 1)    # (T, P, XYZ) -> (P, T, XYZ)
    positions = np.concatenate(positions)
    positions = np.insert(positions, np.s_[::num_timesteps], np.nan, axis = 0)

    radii = np.swapaxes(radii, 0, 1)            # (T, P) -> (P, T)
    radii = np.concatenate(radii)
    radii = np.insert(radii, np.s_[::num_timesteps], np.nan)

    # Compute time spent between consecutive particle positions
    times = np.tile(times, num_particles)
    times = np.insert(times, np.s_[::num_timesteps], np.nan)
    dt = times[1:] - times[:-1]

    # Compute residence distribution in the XZ plane (i.e. granular drum side)
    sim_rtd = kc.dynamic2d(
        positions[:, [0, 2]],
        kc.RATIO,
        values = dt,
        radii = radii,
        resolution = image_shape,
        xlim = xlim,
        ylim = ylim,
        verbose = False,
    )

    # Post-process the NumPy array of pixels within `sim_rtd`
    sim_pix = sim_rtd.pixels

    # Pixellise / discretise the GranuDrum circular outline
    xgrid = np.linspace(xlim[0], xlim[1], image_shape[0])
    ygrid = np.linspace(ylim[0], ylim[1], image_shape[1])

    # Physical coordinates of all pixels
    xx, yy = np.meshgrid(xgrid, ygrid)

    # Remove the GranuDrum's circular outline
    sim_pix[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((11, 11), np.uint8)
    sim_pix2 = cv2.morphologyEx(
        encode_u8(sim_pix),
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Global thresholding + binarisation
    _, sim_pix2 = cv2.threshold(sim_pix2, 10, 255, cv2.THRESH_BINARY)
    sim_pix2[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    # Colour GranuDrum's background in the raw pixellated image
    sim_pix[
        (xx ** 2 + yy ** 2 < trim * granudrum.radius ** 2) &
        (sim_pix == 0.)
    ] = 7 / 255 * sim_pix.max()

    sim_rtd2 = kc.Pixels(sim_pix2, xlim = xlim, ylim = ylim)

    # Return the original and binarised images
    return sim_rtd, sim_rtd2


# Error function for a GranuTools GranuDrum, quantifying the difference
# between an experimental free surface shape (from an image) and a simulated
# one (from an occupancy grid).
#
# The error value represents the difference in the XZ-projected area of the
# granular drum (i.e. the side view) between the GranuDrum image and the
# occupancy grid of the corresponding simulation.
image_path = "granudrum_45rpm_mcc.bmp"

# Rotating drum system dimensions
granudrum = GranuDrum()

# The thresholded GranuDrum image and occupancy plot, as `kc.Pixels`,
# containing only 0s and 1s
trim = 0.6
img_raw, img_post = image_thresh(granudrum, image_path, trim)
sim_raw, sim_post = simulation_thresh(granudrum, img_raw.pixels.shape,
                                      times, radii, positions, trim)

# Pixel physical dimensions, in mm
dx = (granudrum.xlim[1] - granudrum.xlim[0]) / img_post.pixels.shape[0] * 1000
dy = (granudrum.ylim[1] - granudrum.ylim[0]) / img_post.pixels.shape[1] * 1000

# The error is the total different area, i.e. the number of pixels with
# different values times the area of a pixel
error = np.sum(img_post.pixels != sim_post.pixels) * dx * dy


# Plot the simulated and imaged GranuDrums and the difference between them
if show_plots:
    import plotly
    from plotly.subplots import make_subplots

    # Create colorscale starting from white
    cm = plotly.colors.sequential.Blues
    cm[0] = 'rgb(255,255,255)'

    fig = make_subplots(rows = 1, cols = 3)

    # Plot "raw", untrimmed images
    img_raw, img_post = image_thresh(granudrum, image_path, trim = 1.)
    sim_raw, sim_post = simulation_thresh(granudrum, img_post.pixels.shape,
                                          times, radii, positions, trim = 1.)

    # Plot GranuDrum photograph on the left
    fig.add_trace(img_raw.heatmap_trace(colorscale = cm), 1, 1)

    # Plot LIGGGHTS simulation on the right
    fig.add_trace(sim_raw.heatmap_trace(colorscale = cm), 1, 3)

    # Plot both simulation and experiment, colour-coding differences in the
    # middle
    diff = np.zeros(img_raw.pixels.shape)
    diff[(img_post.pixels == 255) & (sim_post.pixels == 255)] = 64
    diff[(img_post.pixels == 255) & (sim_post.pixels == 0)] = 128
    diff[(img_post.pixels == 0) & (sim_post.pixels == 255)] = 192

    diff = kc.Pixels(diff, img_raw.xlim, img_raw.ylim)

    # "Whiten" / blur the areas not used
    xgrid = np.linspace(diff.xlim[0], diff.xlim[1], diff.pixels.shape[0])
    ygrid = np.linspace(diff.ylim[0], diff.ylim[1], diff.pixels.shape[1])
    xx, yy = np.meshgrid(xgrid, ygrid)
    diff.pixels[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] *= 0.2

    fig.add_trace(diff.heatmap_trace(colorscale = cm), 1, 2)

    # Equalise axes
    for i in range(3):
        yaxis = f"yaxis{i + 1}" if i > 0 else "yaxis"
        xaxis = f"x{i + 1}" if i > 0 else "x"
        fig.layout[yaxis].update(scaleanchor = xaxis, scaleratio = 1)

    fig.show()
