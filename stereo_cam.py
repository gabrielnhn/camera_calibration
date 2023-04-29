import numpy as np
import cv2
from matplotlib import pyplot as plt
imgL = cv2.imread('liq1.jpg',0)
imgR = cv2.imread('liq2.jpg',0)
# imgL = cv2.imread('original_r.webp', 0)
# imgR = cv2.imread('original_l.webp', 0)

block_size = 15
stereo = cv2.StereoSGBM.create(

    numDisparities=32,
    blockSize=block_size,
    # texture_threshold=1,
    P1= 8*block_size*block_size,
    P2 = 32*block_size*block_size,
    uniquenessRatio = 5,
    # mode = cv2.STEREO_SGBM_MODE_HH,
)

# 8\*number_of_image_channels\*blockSize\*blockSize and
#  |      .       32\*number_of_image_channels\*blockSize\*blockSize 

disparity = stereo.compute(imgL,imgR)

disparity = disparity.astype(np.uint8)

# plt.imshow(disparity,'gray')
# plt.show()

out = np.hstack((imgL, disparity, imgR))

cv2.imshow("buth", out)
cv2.imwrite("LIQUIDS.png", out)

cv2.waitKey(0)