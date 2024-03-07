import numpy as np
import glob
import os
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import linregress
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, minimize
import math
import emcee


def find_first_two_digits(folder_path):
    """
    Finds the first two digits of all files in a designated folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        A list of the first two digits of all files in the folder.
    """
    files = glob.glob(os.path.join(folder_path, '*'))
    first_two_digits = [os.path.basename(file)[:2] for file in files]
    # first_two_digits = list(map(int, first_two_digits))
    return sorted(first_two_digits)


def find_mass_flow_rate(data, window_size):
    # Get timestamps
    mass_data_time = np.arange(0, len(mass_data), 1 / 100)

    # Create a sliding window of the specified size
    mass_windows = np.lib.stride_tricks.sliding_window_view(data, window_size)
    time_windows = np.lib.stride_tricks.sliding_window_view(mass_data_time, window_size)

    # Compute the R-squared value for each window
    slope_array = []
    intercept_array = []
    r_value_array = []
    for row_in_window, window in enumerate(mass_windows):
        slope, intercept, r_value, _, _ = linregress(time_windows[row], window)
        slope_array.append(slope)
        intercept_array.append(intercept)
        r_value_array.append(r_value)

    # Find the window with the highest R-squared value
    max_r_value_index = np.argmax(r_value_array)

    # Find number of slopes that are less than 0.001
    res = sum(1 for s in slope_array if s <= 0)

    # Extract the data with max_r_value_index
    mass_window = mass_windows[max_r_value_index]
    time_window = time_windows[max_r_value_index]
    mass_flow_rate = slope_array[max_r_value_index]
    percentage_of_low_slopes = res/len(slope_array)

    return mass_window, time_window, mass_flow_rate, percentage_of_low_slopes


def calc_bulk_density(calc_positions, calc_radii, powder_mass, cylinder_radius):
    """
    Calculate the bulk density of the powder bed using the particle positions and radii as well as total particle mass
    and cylinder container dimensions.
    Parameters
    ----------
    calc_positions (np.array): Array of particle positions
    calc_radii (np.array): Array of particle radii
    powder_mass (float): Total mass of the powder in the cylinder
    cylinder_radius (float): Radius of the cylinder

    Returns
    -------
    calculated_bulk_density (float): Calculated bulk density of the powder bed

    """
    # Remove nan particles from arrays
    positions_bd = calc_positions[~np.isnan(calc_positions).any(axis=1)]  # Remove nans
    radii_bd = calc_radii[~np.isnan(calc_radii)]  # Remove nans

    # Add and subtract radii from particle heights
    zpositions_radii_add = np.add(positions_bd[:, 2], radii_bd)
    zpositions_radii_subtract = np.subtract(positions_bd[:, 2], radii_bd)

    # Find index of 10 highest particles+radius and value of lowest particles-radius
    highest_particles_indicies = np.argpartition(zpositions_radii_add, -10)[-10:]

    # Find mean value of the highest particles
    highest_particle_positions = zpositions_radii_add[highest_particles_indicies]
    top = np.mean(highest_particle_positions)

    # Find height of the lowest particle
    bottom = np.nanmin(zpositions_radii_subtract)

    # Calculate bulk density
    length = top - bottom
    total_powder_volume = length*math.pi*cylinder_radius**2  # Volume taken up by powder
    calculated_bulk_density = powder_mass/total_powder_volume

    return calculated_bulk_density


def beverloo_equation(D, C, k, bulk_density, g, d):
    return C*bulk_density*np.sqrt(g)*(D-k*d)**(5/2)

def beverloo_equation_gt(D, C, g, D_min):
    return C*np.sqrt(g)*(D-D_min)**(5/2)


def beverloo_equation_gt_linear(D, C_25, g, D_min):
    return C_25*(np.sqrt(g)**(2/5))*(D-D_min)


# User inputs
data_folder = 'data/23Mar09'
gt_beverloo = True
bulk_density_beverloo = False
flowing_bulk_density_beverloo = False
plot_mass_data = False
plot_mass_flow_rate_data = False
plot_mass_flow_rate_data_gt = False
data_record_rate = 1/100  # Data recording frequency in seconds
particle_data_recording_rate = 1/20  # Particle data recording frequency in seconds
average_particle_diameter = 1.5/1000  # Average particle diameter in metres
window_size = 50
bulk_density = 1000  # kg/m^3
invert_mass_arrays = True
granuflow_tube_diameter = 0.0475  # m

# Find orifice sizes
# orifice_sizes = find_first_two_digits(data_folder)
orifice_sizes_ints = [2, 4, 8, 12, 18, 22, 28]  # [float(numeric_string) for numeric_string in orifice_sizes]

# Load data
# all_mass_data = np.array([])
# start_mass = []
# for i, size in enumerate(orifice_sizes_ints):
#
#     mass_data = np.load(f'{data_folder}/{size}mm_mass.npy')
#     # Invert mass arrays
#     if invert_mass_arrays:
#         start_mass.append(mass_data[0])
#         mass_data = np.subtract(mass_data[0], mass_data)
#
#     all_mass_data = np.vstack((all_mass_data, mass_data)) if i > 0 else mass_data
#
#
# # Extract mass flow rate
# all_mass_windows = np.array([])
# all_time_windows = np.array([])
# all_mass_flow_rates = []
# all_percentage_zero = []
# for row, mass_data in enumerate(all_mass_data):
#     mass_window, time_window, mass_flow_rate, percentage_zero = find_mass_flow_rate(mass_data, window_size)
#     all_mass_windows = np.vstack((all_mass_windows, mass_window)) if row > 0 else mass_window
#     all_time_windows = np.vstack((all_time_windows, time_window)) if row > 0 else time_window
#     if mass_data[-1]/start_mass[row] < 0.2 < percentage_zero:
#         mass_flow_rate = float(0)
#     all_mass_flow_rates.append(mass_flow_rate)
#     all_percentage_zero.append(percentage_zero)

# Fit data to the Beverloo equation
D_sizes = np.divide(np.array(orifice_sizes_ints), 1000)
# mass_flow_rates_array = np.array(all_mass_flow_rates)
mass_flow_rates_array = np.divide([0, 0, 2.94, 7.51, 20.98, 33.35, 62.37], 1000)  # Experimental results
upper_bound = np.min(D_sizes)/average_particle_diameter  # Set the upper bound of k to be D/d

# Scipy
# popt, _ = curve_fit(f=beverloo_equation(density=bulk_density, g=9.81, d=average_particle_diameter), xdata=D_sizes,  ydata=mass_flow_rates_array, bounds=(0.0001, upper_bound))
# C, k = popt

if bulk_density_beverloo is True:
    # Lmfit
    powder_bulk_density = np.load(f'{data_folder}/bulk_density.npy')
    model_bulk_dens = Model(beverloo_equation, independent_vars=['D'], param_names=['C', 'k', 'bulk_density', 'g', 'd'])
    params = Parameters()
    params.add('C', value=0.5, min=0, max=2)
    params.add('k', value=1, min=0.001, max=upper_bound)
    params.add('bulk_density', value=powder_bulk_density, vary=False)
    params.add('g', value=9.81, vary=False)
    params.add('d', value=average_particle_diameter, vary=False)

    fit = model_bulk_dens.fit(mass_flow_rates_array, params=params, D=D_sizes)
    test = fit.fit_report()
    C_bulk_beverloo = fit.params['C'].value
    k_bulk_beverloo = fit.params['k'].value

    # Print Results
    print(f'Beverloo C term: {C_bulk_beverloo}')
    print(f'Beverloo k term: {k_bulk_beverloo}')

if flowing_bulk_density_beverloo is True:
    # Calculate flowing bulk density
    particle_positions = np.load(f'{data_folder}/{orifice_sizes_ints[-1]}mm_positions.npy', allow_pickle=True)
    bed_height_array = []
    for i, position_array in enumerate(particle_positions):
        positions_no_nans = position_array[~np.isnan(position_array).any(axis=1)]  # Remove nans
        highest_particles_index = np.argpartition(positions_no_nans[:, 2], -10)[-10:]
        top_particle_positions = positions_no_nans[highest_particles_index]
        height = np.median(top_particle_positions[:, 2])
        bed_height_array.append(height)

    change_in_height = np.diff(bed_height_array)  # Find the difference in height
    max_change_in_height = np.max(change_in_height*-1)  # Find the maximum change in height
    bed_velocity = max_change_in_height/particle_data_recording_rate  # Find the bed velocity in m/s
    bulk_flowing_density = (4*mass_flow_rates_array[-1])/(math.pi*(granuflow_tube_diameter**2)*bed_velocity)  # Find the flowing bulk density

    # Lmfit
    model_bulk_dens = Model(beverloo_equation, independent_vars=['D'], param_names=['C', 'k', 'bulk_density', 'g', 'd'])
    params = Parameters()
    params.add('C', value=0.5, min=0.001, max=2)
    params.add('k', value=1, min=0.001, max=upper_bound)
    params.add('bulk_density', value=bulk_flowing_density, vary=False)
    params.add('g', value=9.81, vary=False)
    params.add('d', value=average_particle_diameter, vary=False)

    fit = model_bulk_dens.fit(mass_flow_rates_array, params=params, D=D_sizes)
    test = fit.fit_report()
    C_bulk_beverloo = fit.params['C'].value
    k_bulk_beverloo = fit.params['k'].value

    # Print Results
    print(f'Beverloo C term: {C_bulk_beverloo}')
    print(f'Beverloo k term: {k_bulk_beverloo}')

# if gt_beverloo is True:
#     # Lmfit Granutools Beverloo Fit
#     model = Model(beverloo_equation_gt, independent_vars=['D'], param_names=['C', 'g', 'D_min'])
#     params = Parameters()
#     params.add('C', value=0.5, min=0, max=4)
#     minimum_D_flow = D_sizes[np.argmax(np.where(mass_flow_rates_array==0)[0])+1]
#     D_sizes_gt = D_sizes[np.where(mass_flow_rates_array!=0)]*1000
#     params.add('D_min', value=0.1, min=0, max=minimum_D_flow*1000)
#     params.add('g', value=9810, vary=False)
#
#     mass_flow_rates_array_gt = mass_flow_rates_array[np.where(mass_flow_rates_array!=0)]*1000
#
#     fit = model.fit(mass_flow_rates_array_gt, params=params, D=D_sizes_gt, method='emcee')
#     test=fit.fit_report()
#     C_gt_beverloo = fit.params['C'].value
#     D_min_gt_beverloo = fit.params['D_min'].value
#
#     # Print Results
#     print(f'Beverloo C term: {C_gt_beverloo}')
#     print(f'Beverloo D_min term: {D_min_gt_beverloo}')

if gt_beverloo is True:
    # Lmfit Granutools Beverloo Linear Fit
    mass_flow_rates_array_25 = np.power(mass_flow_rates_array*1000, 2/5)
    D_sizes_gt = D_sizes[np.where(mass_flow_rates_array != 0)] * 1000

    model = Model(beverloo_equation_gt_linear, independent_vars=['D'], param_names=['C_25', 'g', 'D_min'])
    params = Parameters()
    params.add('C_25', value=0.5, min=0, max=4)
    params.add('D_min', value=0.1, min=0, max=10)
    params.add('g', value=9810, vary=False)

    mass_flow_rates_array_gt_25 = mass_flow_rates_array_25[np.where(mass_flow_rates_array_25!=0)]

    fit = model.fit(mass_flow_rates_array_gt_25, params=params, D=D_sizes_gt, method='emcee')
    test = fit.fit_report()
    C_gt_beverloo_25 = fit.params['C_25'].value
    D_min_gt_beverloo = fit.params['D_min'].value

    C_gt_beverloo = C_gt_beverloo_25**(5/2)

    # Print Results
    print(f'Beverloo C term: {C_gt_beverloo}')
    print(f'Beverloo D_min term: {D_min_gt_beverloo}')

# Plot data
if plot_mass_data is True:
    # Plot graph
    fig = go.Figure()
    colours = sns.color_palette("colorblind")
    colours = colours.as_hex()
    for row, mass_data in enumerate(all_mass_data):
        x_time = np.arange(0, len(mass_data), 1/100)
        fig.add_trace(go.Scatter(x=x_time,
                                 y=mass_data,
                                 mode='markers', # or lines
                                 marker=dict(size=8, color=colours[row]),  # Change colours value per trace added
                                 name=f'Orifice Size: {orifice_sizes[row]}mm'
                                 )
                      )
        fig.add_trace(go.Scatter(x=all_time_windows[row],
                                 y=all_mass_windows[row],
                                 mode='markers',  # or lines
                                 marker=dict(size=8, color='black'),  # Change colours value per trace added
                                 name=f'Sliding window: {orifice_sizes[row]}mm'
                                 )
                      )

    fig.update_layout(
        title_text="Mass that has left the GranuFlow over time",
        title_font_size=30,
        legend=dict(font=dict(size=20)))
    fig.update_xaxes(
            title_text="Time (sec)",
            title_font={"size": 20},
            tickfont_size=20)
    fig.update_yaxes(
            title_text="Mass (kg)",
            title_font={"size": 20},
            tickfont_size=20)
    fig.show()

if plot_mass_flow_rate_data and bulk_density_beverloo is True:
    # Plot graph
    fig = go.Figure()
    colours = sns.color_palette("colorblind")
    colours = colours.as_hex()
    all_mass_flow_rates_52 = np.power(all_mass_flow_rates, 2/5)

    curve_d_sizes = np.arange(orifice_sizes_ints[0], orifice_sizes_ints[-1], 0.1)

    fig.add_trace(go.Scatter(x=orifice_sizes_ints,
                             y=all_mass_flow_rates,
                             mode='markers',  # or lines
                             marker=dict(size=15, color=colours[0]),  # Change colours value per trace added
                             name=f'Mass Flow Rate'
                             )
                  )
    fig.add_trace(go.Scatter(x=curve_d_sizes,
                             y=beverloo_equation(np.divide(curve_d_sizes, 1000), C_bulk_beverloo, k_bulk_beverloo, bulk_density=bulk_density, g=9.81, d=average_particle_diameter),
                             mode='lines',
                             line=dict(color=colours[1], width=3),
                             name=f'Beverloo Equation'
                             )
                  )

    fig.update_layout(
        title_text="Mass flow rate at each orifice size in the GranuFlow",
        title_font_size=30,
        legend=dict(font=dict(size=20)))
    fig.update_xaxes(
        title_text="Orifice Size (mm)",
        title_font={"size": 20},
        tickfont_size=20)
    fig.update_yaxes(
        title_text="Mass flow rate (kg/s)",
        title_font={"size": 20},
        tickfont_size=20)
    fig.show()

if plot_mass_flow_rate_data_gt and gt_beverloo is True:
    # Plot graph
    fig = go.Figure()
    colours = sns.color_palette("colorblind")
    colours = colours.as_hex()
    all_mass_flow_rates_52 = np.power(all_mass_flow_rates, 2/5)*1000

    curve_d_sizes = np.arange(D_min_gt_beverloo, D_sizes_gt[-1], 0.1)

    fig.add_trace(go.Scatter(x=orifice_sizes_ints,
                             y=mass_flow_rates_array*1000,
                             mode='markers',  # or lines
                             marker=dict(size=15, color=colours[0]),  # Change colours value per trace added
                             name=f'Mass Flow Rate'
                             )
                  )
    fig.add_trace(go.Scatter(x=curve_d_sizes,
                             y=beverloo_equation_gt(curve_d_sizes, C_gt_beverloo, g=9810, D_min=D_min_gt_beverloo),
                             mode='lines',
                             line=dict(color=colours[1], width=3),
                             name=f'Beverloo Equation'
                             )
                  )

    fig.update_layout(
        title_text="Mass flow rate at each orifice size in the GranuFlow",
        title_font_size=30,
        legend=dict(font=dict(size=20)))
    fig.update_xaxes(
        title_text="Orifice Size (mm)",
        title_font={"size": 20},
        tickfont_size=20)
    fig.update_yaxes(
        title_text="Mass flow rate (kg/s)",
        title_font={"size": 20},
        tickfont_size=20)
    fig.show()

plot_gt_linear = True
if plot_gt_linear is True:
    # Plot graph
    fig = go.Figure()
    colours = sns.color_palette("colorblind")
    colours = colours.as_hex()

    curve_d_sizes = np.arange(D_min_gt_beverloo, D_sizes_gt[-1], 0.1)

    fig.add_trace(go.Scatter(x=orifice_sizes_ints,
                             y=mass_flow_rates_array_25,
                             mode='markers',  # or lines
                             marker=dict(size=15, color=colours[0]),  # Change colours value per trace added
                             name=f'Mass Flow Rate'
                             )
                  )
    fig.add_trace(go.Scatter(x=curve_d_sizes,
                             y=beverloo_equation_gt_linear(curve_d_sizes, C_gt_beverloo_25, g=9810, D_min=D_min_gt_beverloo),
                             mode='lines',
                             line=dict(color=colours[1], width=3),
                             name=f'Beverloo Equation'
                             )
                  )

    fig.update_layout(
        title_text="Mass flow rate at each orifice size in the GranuFlow",
        title_font_size=40,
        legend=dict(font=dict(size=20)))
    fig.update_xaxes(
        title_text="Orifice Size (mm)",
        title_font={"size": 30},
        tickfont_size=20)
    fig.update_yaxes(
        title_text="Mass flow rate<sup>2/5</sup> (kg/s)<sup>2/5</sup>",
        title_font={"size": 30},
        tickfont_size=20)
    fig.show()