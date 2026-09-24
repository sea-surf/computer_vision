import numpy as np
import cv2 as cv

if __name__ == "__main__":
    # Invert the x-ray image to make the bones appear white and the background black

    # Inverter a cor dos ossos de cinza para branco e o fundo de branco para preto
    img = cv.imread('../data/img/xray-inverted.jpg')
    height, width, _ = img.shape

    # invert color from the image: white - darker color = whiter color
    img = 255 - img
    cv.imwrite("../data/img/xray.jpg", img)