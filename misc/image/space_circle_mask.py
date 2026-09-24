import numpy as np
import cv2 as cv
import random

if __name__ == "__main__":
    
    space = cv.imread("../data/img/space.jpg")

    # Apply grayscale to the image
    gray = cv.cvtColor(space, cv.COLOR_BGR2GRAY)

    # Everything greater than 50 transform to 255, everything else to black 0
    _, mask = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    # _, mask = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)

    cv.imwrite("../data/img/mask.jpg", mask)

    # Every white point on my original image 
    space[mask == 255] = (255, 0, 0)

    cv.imwrite("../data/img/space_masked.jpg", space)

    # -------------------------

    circle = cv.imread("../data/img/circle.png")

    gray_circle = cv.cvtColor(circle, cv.COLOR_BGR2GRAY)

    cv.imwrite("../data/img/gray_circle.jpg", gray_circle)

    thresh, mask_circle = cv.threshold(gray_circle, 50, 255, cv.THRESH_OTSU)
    print(f'Thresh: {thresh}')
    cv.imwrite("../data/img/mask_circle.jpg", mask_circle)

    # -------------------