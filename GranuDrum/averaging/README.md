# Averaging GranuDrum Images
The averaged_image.py file takes the .bmp images that the GranuDrum outputs and averages them into a single image that can be read into ACCES. The image of the free surface shape is used for comparison against the LIGGGHTS simulation.

## Dependencies
- cv2: "import cv2 as cv"
- numpy; "import numpy as np"
- pept: "import pept"
- glob: "from glob import glob"
- PLotlyGrapher2D from the pept library needed: "from pept.plots import PlotlyGrapher2D"


## Example Run
python averaged_image.py
