import numpy as np
import cv2 as cv
import random

if __name__ == "__main__":
    
    # Threshhold space using Otsu
    space = cv.imread("../data/img/space.jpg")

    space_gray = cv.cvtColor(space, cv.COLOR_BGR2GRAY)

    cv.imwrite("../data/img/space_gray.jpg", space_gray)

    _, mask = cv.threshold(space_gray, 50, 255, cv.THRESH_OTSU)

    cv.imwrite("../data/img/space_mask_otsu.jpg", mask)