import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('one.jpeg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('other.jpeg', cv.IMREAD_GRAYSCALE)


# stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
stereo = cv.StereoBM_create(numDisparities=256, blockSize=11)
disparity = stereo.compute(imgL,imgR)
# cv.imshow('title', disparity)
# cv.waitKey(0)

plt.imshow(disparity, "gray")
plt.show()