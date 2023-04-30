import cv2
import numpy as np
import pickle
from datetime import datetime as dt
import glob

calib_cam = pickle.load(open("calib_cam.pkl", "rb"))

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    mtx = calib_cam["mtx"]
    dist = calib_cam["dist"]

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]

    output = np.vstack((img, dst))

    cv2.imshow("BRUH", output)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite(f"undistorted{dt.now()}.png", output)
