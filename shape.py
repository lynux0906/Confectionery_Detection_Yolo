import cv2
import numpy as np

img = cv2.imread('oreo.jpg')
img_size = cv2.resize(img, None, fx = 0.5, fy = 0.5)
height, width, channels = img.shape

print(img.shape)
cv2.imshow('Image', img_size)
cv2.waitKey(0)
cv2.destroyAllWindows()
