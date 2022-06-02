import os
import pathlib
import cv2
import numpy as np
import operator
from datetime import datetime
from shutil import copyfile
import xml.etree.ElementTree as ET
import sys
import math
from numpy.core.fromnumeric import var

sys.path.insert(1,'./deep-person-reid/')
import torch
import torchreid.utils
