import cv2
import glob
import numpy as np

imgs = []
for fname in glob.glob('./results/*.jpg'):
    imgs.append(cv2.imread(fname))

output = np.hstack((imgs[0], imgs[1], imgs[2]))
for i in range(3, 13, 3):
    more_imgs = np.hstack((imgs[i], imgs[i + 1], imgs[i + 2]))
    output = np.vstack((output, more_imgs))

cv2.imshow('output', output)
cv2.waitKey(0)
cv2.imwrite('RESULTS.png', output)

