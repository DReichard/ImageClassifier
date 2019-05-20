import math
import os
import sys
import ntpath
import tempfile
import subprocess

import numpy as np
from scipy import ndimage, misc
from cmath import sqrt

from PIL import Image
from PIL.ExifTags import TAGS

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

found = multiprocessing.Value('i', 0)


# -- APPENDED FILES --

def extra_size(filename):
    print("WARNING! not implemented")

    name = ntpath.basename(filename)
    I = misc.imread(filename)

    misc.imsave(tempfile.gettempdir() + '/' + name, I)
    return 0


# -- EXIF --

# {{{ exif()
def exif(filename):
    image = Image.open(filename)
    try:
        exif = {TAGS[k]: v for k, v in image._getexif().items() if k in TAGS}
        return exif

    except AttributeError:
        return {}


# }}}


# -- SAMPLE PAIR ATTACK --

# {{{ spa()
"""
    Sample Pair Analysis attack. 
    Return Beta, the detected embedding rate.
"""


def spa(filename, channel=0):
    if channel != None:
        I3d = misc.imread(filename)
        width, height, channels = I3d.shape
        I = I3d[:, :, channel]
    else:
        I = misc.imread(filename)
        width, height = I.shape

    x = 0;
    y = 0;
    k = 0
    for j in range(height):
        for i in range(width - 1):
            r = I[i, j]
            s = I[i + 1, j]
            if (s % 2 == 0 and r < s) or (s % 2 == 1 and r > s):
                x += 1
            if (s % 2 == 0 and r > s) or (s % 2 == 1 and r < s):
                y += 1
            if round(s / 2) == round(r / 2):
                k += 1

    if k == 0:
        print("ERROR")
        sys.exit(0)

    a = 2 * k
    b = 2 * (2 * x - width * (height - 1))
    c = y - x

    bp = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    bm = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    beta = min(bp.real, bm.real)
    return beta


# }}}


# -- RS ATTACK --

# {{{ solve()
def solve(a, b, c):
    sq = np.sqrt(b ** 2 - 4 * a * c)
    return (-b + sq) / (2 * a), (-b - sq) / (2 * a)


# }}}

# {{{ smoothness()
def smoothness(I):
    return (np.sum(np.abs(I[:-1, :] - I[1:, :])) +
            np.sum(np.abs(I[:, :-1] - I[:, 1:])))


# }}}

# {{{ groups()
def groups(I, mask):
    grp = []
    m, n = I.shape
    x, y = np.abs(mask).shape
    for i in range(m - x):
        for j in range(n - y):
            grp.append(I[i:(i + x), j:(j + y)])
    return grp


# }}}

# {{{ difference()
def difference(I, mask):
    cmask = - mask
    cmask[(mask > 0)] = 0
    L = []
    for g in groups(I, mask):
        flip = (g + cmask) ^ np.abs(mask) - cmask
        L.append(np.sign(smoothness(flip) - smoothness(g)))
    N = len(L)
    R = float(L.count(1)) / N
    S = float(L.count(-1)) / N
    return R - S


# }}}

# {{{ rs()
def rs(I):
    try:
        I = I.astype(int)

        mask = np.array([[1, 0], [0, 1]])
        d0 = difference(I, mask)
        d1 = difference(I ^ 1, mask)

        mask = -mask
        n_d0 = difference(I, mask)
        n_d1 = difference(I ^ 1, mask)

        p0, p1 = solve(2 * (d1 + d0), (n_d0 - n_d1 - d1 - 3 * d0), (d0 - n_d0))
        if np.abs(p0) < np.abs(p1):
            z = p0
        else:
            z = p1

        if math.isnan(z):
            return 0.0
        if abs(z / (z - 0.5)) > 1:
            return 0.0
        return abs(z / (z - 0.5))
    except:
        return 0.0





