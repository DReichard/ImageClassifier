import base64
import io
import json

import matplotlib
import skimage.io as skio
import cv2
from scipy import misc
import numpy as np
from PIL import Image

target_path = r"D:\diploma\gallery\test\1.jpg"

I = cv2.imread(target_path, 0).flatten()
with open(r"D:\diploma\gallery\test\js_cv.json", 'r') as f:
    cv2image = json.load(f)
with open(r"D:\diploma\gallery\test\js_all.json", 'r') as f:
    jsimage = json.load(f)

print("I length: " + str(len(I)))
print("js length: " + str(len(jsimage)))
print("js cv length: " + str(len(cv2image)))
k = 4
del jsimage[k-1::k]
print("jsfiltered length: " + str(len(jsimage)))
print("cv2 length: " + str(len(cv2image)))
diff = 0
for x,y in zip(cv2image, I):
    if x != y:
        diff += 1
        print("difference: " + str(x - y))
print("total: " + str(diff/len(cv2image)) + "%")
