#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : psd_fractions.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 27.05.2021


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objs as go


# Number-based diameters, given in um
psd = pd.read_csv("solidsizer_mcc.csv").to_numpy()
diameters_particles = psd[:, 1]
intensity, diameters = np.histogram(diameters_particles, 20, [900, 1600])

diameters = (diameters[1:] + diameters[:-1]) / 2
probability = interp1d(diameters, intensity / intensity.sum(), "cubic")

fractions_diameters = np.array([
    1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500
])
fractions_probability = probability(fractions_diameters)
fractions_probability /= fractions_probability.sum()

print(f"Diameters:\n{fractions_diameters} um")
print(f"Fractions:\n{fractions_probability}\n\n")

# Calculate approximate number of coarse-grained, randomly packed particles
cg_factor = 1
packing = 0.3
volume = 50e-6      # GranuDrum fill volume: 50 mL -> 50e-6 L

spheres_volume = np.pi / 6 * (1e-6 * fractions_diameters * cg_factor) ** 3
bulk_volume = (spheres_volume * fractions_probability).sum() / packing
num_particles = volume / bulk_volume

density = 1580
mass = (np.round(num_particles * fractions_probability) *
        spheres_volume).sum() * density

print(f"Coarse-grained (factor = {cg_factor}) diameters:\n",
      f"{fractions_diameters * cg_factor} um")
print(f"Approximate number of particles in {volume * 1e6} mL: {num_particles}")
print(f"Equivalent to {mass * 1000} g")


# Plot the particle size distribution and the selected fractions
fig = go.Figure()

width = (diameters[-1] - diameters[0]) / (diameters.shape[0] - 1)
continuous = np.linspace(diameters[0], diameters[-1], 100)

fig.add_trace(
    go.Bar(
        name = "Solidsizer Measurements",
        x = diameters,
        y = probability(diameters),
        width = width,
    )
)
fig.add_trace(
    go.Scatter(
        name = "Continuous Interpolation",
        x = continuous,
        y = probability(continuous),
    )
)
fig.add_trace(
    go.Bar(
        name = "Extracted Fractions",
        x = fractions_diameters,
        y = probability(fractions_diameters),
        width = width / 5,
        marker = dict(color = "red"),
    )
)


def format_fig(fig):
    '''Format a Plotly figure to a consistent theme for the Nature
    Computational Science journal.'''

    # LaTeX font
    font = "Computer Modern"
    size = 20
    fig.update_layout(
        font_family=font,
        font_size=size,
        title_font_family=font,
        title_font_size=size,
    )

    for an in fig.layout.annotations:
        an["font"]["size"] = size

    fig.update_layout(
        xaxis = dict(title_font_family=font, title_font_size=size),
        yaxis = dict(title_font_family=font, title_font_size=size),
        legend = dict(yanchor="top", xanchor="right"),
    )

    fig.update_layout(template = "plotly_white")


fig.update_layout(
    xaxis_title = "Diameter (μm)",
    yaxis_title = "Probability",
)
format_fig(fig)

fig.show()

fig.write_image("psd_fractions.png", height = 600, width = 800)
