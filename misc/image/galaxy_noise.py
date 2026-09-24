import numpy as np
import cv2 as cv
import random

if __name__ == "__main__":
    
    img = cv.imread("../data/img/space.jpg")

    colors = [(205,211,194), (169,164,159), (76,73,86)]

    height, width, _ = img.shape

    for h in range(height):
        for w in range(width):
            pixel = img[h][w]
            if pixel[0] < 30 and pixel[1] < 30 and pixel[2] < 30: 
                color = random.randint(0,2)
                img[h][w] = colors[color]
                # choose a random color for each pixel
    
    cv.imwrite("../data/img/space_noise.jpg", img)


    # RGB != BGR opencv
    # [(194,211,205), (159,164,169), (86,73,76)]