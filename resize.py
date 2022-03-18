import os, glob
import cv2
import numpy as np
import uuid

img = cv2.imread('F:\Python\XLA\yolo\oreo.jpg')

img = cv2.resize(img, None, fx = 1/3, fy = 1/3)

cv2.imwrite('F:\Python\XLA\yolo\oreo1.jpg', img)

