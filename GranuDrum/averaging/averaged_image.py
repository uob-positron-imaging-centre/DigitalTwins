#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : averaged_image.py
# License: GNU v3.0
# Author : Jack Sykes <jas653@student.bham.ac.uk>
# Date   : 11.03.2022

import cv2 as cv
import numpy as np
import pept
from glob import glob
from pept.plots import PlotlyGrapher2D


# insert the filepath where the GranuDrum .bmp files are located
files = glob(r".../*_Frame.bmp")
image = np.float64(255 - cv.imread(files[0], cv.IMREAD_GRAYSCALE)[::-1].T)

for f in files[1:]:
    image += 255 - cv.imread(f, cv.IMREAD_GRAYSCALE)[::-1].T

image /= len(files)

# save the image with an appropriate name
cv.imwrite("gd_averaged.bmp", 255 - image.T[::-1])

pixels = pept.Pixels(image, xlim = [-0.042, 0.042], ylim = [-0.042, 0.042])

grapher = PlotlyGrapher2D().add_pixels(pixels)
grapher.fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
grapher.show()
