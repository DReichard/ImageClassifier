import cv2
import numpy as np
from numpy import array, int32


def pgmread(filename):
    """  This function reads Portable GrayMap (PGM) image files and returns
  a numpy array. Image needs to have P2 or P5 header number.
  Line1 : MagicNum
  Line2 : Width Height
  Line3 : Max Gray level
  Lines starting with # are ignored """
    img = cv2.imread(filename, 0)
    f = np.float32(img)
    return array(f)


def pgmwrite(img, filename, maxVal=255, magicNum='P5'):
    """  This function writes a numpy array to a Portable GrayMap (PGM)
  image file. By default, header number P2 and max gray level 255 are
  written. Width and height are same as the size of the given list.
  Line1 : MagicNum
  Line2 : Width Height
  Line3 : Max Gray level
  Image Row 1
  Image Row 2 etc. """
    img = int32(img).tolist()
    f = open(filename, 'w')
    width = 0
    height = 0
    for row in img:
        height = height + 1
        width = len(row)
    f.write(magicNum + '\n')
    f.write(str(width) + ' ' + str(height) + '\n')
    f.write(str(maxVal) + '\n')
    for i in range(height):
        count = 1
        for j in range(width):
            f.write(str(img[i][j]) + ' ')
            if count >= 17:
                # No line should contain gt 70 chars (17*4=68)
                # Max three chars for pixel plus one space
                count = 1
                f.write('\n')
            else:
                count = count + 1
        f.write('\n')
    f.close()
