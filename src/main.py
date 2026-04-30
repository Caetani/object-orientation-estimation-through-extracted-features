import pandas as pd
import json
import numpy as np

from utils.orientation_utils import *

if __name__ == '__main__':
    r = euler_to_quaternion(0, 0, 0)
    print(r)