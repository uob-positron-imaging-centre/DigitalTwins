#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : occupancy.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 25.01.2022


'''Given a 3D array of particle positions (Timestep, Particle, XYZ), generate
a static 2D occupancy plot - this corresponds to taking an instantaneous X-Ray
image of the particles from a side.
'''


import numpy as np
import konigcell as kc

import plotly.graph_objs as go


# Generate some particle positions in the same format as outputted by the
# PICI-LIGGGHTS Python interface
num_timesteps = 10
num_particles = 5

positions = np.random.normal(size=(num_timesteps, num_particles, 3))
radii = np.random.normal(size=(num_timesteps, num_particles))


# Extract XZ coordinates and concatenate them across all timesteps (axis 0)
positions2d = positions[:, :, [0, 2]]                   # Use [0, 1] for XY
positions2d = np.concatenate(positions2d, axis=0)
radii2d = np.concatenate(radii, axis=0)

occupancy = kc.static2d(
    positions2d,
    kc.INTERSECTION,
    radii=radii2d,
    resolution=(500, 500),
    # xlim=[xmin, xmax],
    # ylim=[ymin, ymax],
)


# Plot occupancy grid as a heatmap - i.e. greyscale image
fig = go.Figure()
fig.add_trace(occupancy.heatmap_trace())

fig.update_layout(
    title="Occupancy Grid",
    xaxis_title="x (mm)",
    yaxis_title="z (mm)",

    yaxis_scaleanchor = "x",
    yaxis_scaleratio = 1,

    template="plotly_white",
)
fig.show()
