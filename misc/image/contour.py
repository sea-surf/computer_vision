import numpy as np
import cv2 as cv

if __name__ == "__main__":
    
    img = cv.imread("../data/img/patent.jpg")
    
    # guarantees the img is in black and white
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # <20 dark -> darker >20 dark -> white
    _, th = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)

    cv.imwrite("../data/img/patent_th.jpg", th)
    
    # extract the contour of the image
    contours, hierarchy = cv.findContours(
        th, 
        cv.RETR_EXTERNAL, 
        cv.CHAIN_APPROX_SIMPLE
    )

    # print(f'Contornos: {contours}')

    # original img in the end, make a copy to not modify the original img
    drawn = cv.drawContours(img.copy(), contours, -1, (0,255,0))

    cv.imwrite("../data/img/patent_contour.jpg", drawn)

    # filter the contours by area, only keep the ones with area > 200
    contours_filtered = []

    for contour in contours:
        #print(cv.contourArea(contour))
        if cv.contourArea(contour) > 200:
            contours_filtered.append(contour)

    drawn_filtered = cv.drawContours(img, contours_filtered, -1, (0,255,0))

    cv.imwrite("../data/img/patent_contour_filtered.jpg", drawn_filtered)