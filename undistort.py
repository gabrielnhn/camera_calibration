import cv2
import numpy as np
import pickle

calib_cam = pickle.load(open("calib_cam.pkl", "rb"))


capture = cv2.VideoCapture(2)

retval, img = capture.read()

while retval:
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
    cv2.waitKey(3)

    retval, img = capture.read()

