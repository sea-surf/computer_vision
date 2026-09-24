import numpy as np
import cv2 as cv
import random

if __name__ == "__main__":
    
    space = cv.imread("../data/img/space_noise.jpg")
    
    # space_blur = cv.GaussianBlur(space, (3, 3), 100)
    space_blur = cv.medianBlur(space, 5)

    cv.imwrite("../data/img/space_blur.jpg", space_blur)

    space_blur_gray = cv.cvtColor(space_blur, cv.COLOR_BGR2GRAY)
    _, background_mask = cv.threshold(space_blur_gray, 50, 255, cv.THRESH_OTSU)

    cv.imwrite("../data/img/space_blur_otsu.jpg", background_mask)

    # Sem blur

    space_gray = cv.cvtColor(space, cv.COLOR_BGR2GRAY)
    _, background_mask_no_blur = cv.threshold(space_gray, 50, 255, cv.THRESH_OTSU)
    cv.imwrite("../data/img/space_blurfalse_otsu.jpg", background_mask_no_blur)