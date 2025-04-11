#################
# 工程所需的库和模块
#################
import os
import glob
import sys
import random
import imageio
import webbrowser
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from matplotlib.ticker import LinearLocator
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from natsort import natsorted