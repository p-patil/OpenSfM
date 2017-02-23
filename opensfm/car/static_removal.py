import numpy as np
import cv2
cap = cv2.VideoCapture('/Users/yang/Downloads/visual odometry/data/incident-92db210c1997e92fb1ad5d12799636f9.mov')
# This is not working yet, since static part usualy include road and sky as well.
fgbg =  cv2.BackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
