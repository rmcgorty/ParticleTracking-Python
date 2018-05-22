import numpy as np
from trackmem import trackmem
import os

#added argument mt=None (so doesn't have to read file), also fovn is now optional
def run(basepath, featuresize, mt=None, fovn=0, maxdisp=2, goodenough=2, memory=1):
    # REVISION HISTORY
    # written by Paul Fournier and Vincent Pelletier (Maria Kilfoil's group),
    # July 2013 - Adapted for Python by Kevin Smith (Maria Kilfoil's group)
    #
    # A little shell that runs trackmem on the output from mpretrack.run. 
    #
    # INPUTS
    #
    # basepath    - the basepath of the experiments. Reads the MT matrix from
    #              "fov#\\MT_featsize_#.npy", as output by mpretrack
    # fovn        - specifies which series of images to process
    # featuresize - specifies the feature size for accessing the right MT file
    # maxdisp     - (optional) specifies the maximum displacement (in pixels) a feature may 
    #              make between successive frames
    # goodenough  - (optional) the minimum length requirement for a trajectory to be retained 
    # memory      - (optional) specifies how many consecutive frames a feature is allowed to skip. 
    #
    # OUTPUTS
    #
    # saves the tracks as Tracks_featsize_#.npy in the same folder as 
    # MT_featsize_#.npy
    #
    # tracks matrix format:
    # 1 row per bead per frame, sorted by bead ID then by frame number.
    # columns are:
    # 0:1 - X and Y positions (in pixels)
    # 2   - Integrated intensity
    # 3   - Rg squared of feature
    # 4   - eccentricity
    # 5   - frame #basepath
    # 6   - time of frame
    # 7   - Bead ID
    
    MT = np.load(os.path.join(basepath,"fov" + str(fovn), "MT_featsize_" + str(featuresize) + ".npy"))
    tracks = trackmem(MT, maxdisp, 2, goodenough, memory)
    np.save(os.path.join(basepath,"fov" + str(fovn), "Tracks_featsize_" + str(featuresize) + ".npy"),tracks)
