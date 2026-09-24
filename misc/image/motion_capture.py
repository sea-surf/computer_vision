import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame1 = None
frame2 = None

while True:

    frame2 = frame1
    ret, frame1 = cap.read()

    # apply grayscale on frame1
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    if frame2 is None:
        continue    

    # Subtract the difference between frames
    diff = cv.absdiff(frame1, frame2)

    cv.imshow("Motion Capture", diff)

    if cv.waitKey(30) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()