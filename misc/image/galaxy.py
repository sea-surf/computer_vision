import numpy as np
import cv2 as cv

if __name__ == "__main__":
    
    img = cv.imread("../data/img/space.jpg")
    height, width, _ = img.shape
    for h in range(height):
        for w in range(width):
            pixel = img[h][w]
            if pixel[0] > 200: 
                img[h][w] = (0, 0, 255)
                # turn blue color pixels (~255,0,0) into red (0,0,~255)
    
    cv.imwrite("../data/img/space3.jpg", img)


    # Você está tentando fazer uma peça artistica beseado nas imagens do James Webb Telescope.
    # Você deve primeiro extrair o fundo preto da imagem e substitui-lo por uma das três cores em RGB aleatoriamente:
    # [(194,211,205), (159,164,169), (86,73,76)]

    # Nota: Lembre-se que por padrão o opencv abre a imagem em BGR em vez de RGB!