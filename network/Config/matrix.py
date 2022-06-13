#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import cos, sin, pi

SCENE_ROT = np.array([
    [1, 0, 0, 0],
    [0, cos(pi), -sin(pi), 0],
    [0, sin(pi), cos(pi), 0],
    [0, 0, 0, 1]
])

SCENE_ROT_INV = np.linalg.inv(SCENE_ROT)

