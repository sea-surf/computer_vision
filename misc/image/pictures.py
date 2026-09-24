import numpy as np
import cv2 as cv

if __name__ == "__main__":

    img = cv.imread('../data/img/earth.jpg')

    # turn the image to grayscale and save it
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(gray.shape)
    cv.imwrite("../data/img/earth_gray.jpg", gray)


    

    """
    Additional code for image manipulation:
    
    print(f'Shape (H, W, C): {img.shape}')
    print(f'Max Pixel Value: {img.max()}')
    print(f'Min Pixel Value: {img.min()}')

    # set regions of the image blue and green to 0, leaving only red
    img[:,:,0] = 0
    img[:,:,1] = 0
    cv.imwrite("../data/img/earth2.jpg", img)

    # turn the image to grayscale and save it
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(gray.shape)
    cv.imwrite("../data/img/earth_gray.jpg", gray)

    # turn darker pixels into white
    height, width, _ = img.shape
    for h in range(height):
        for w in range(width):
            pixel = img[h][w]
            if pixel[0] < 50 and pixel[1] < 50 and pixel[2] < 50:
                img[h][w] = (255, 255, 255)

    cv.imwrite("../data/img/earth3.jpg", img)
    """
