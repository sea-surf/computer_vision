import numpy as np
import cv2 as cv
from collections import defaultdict

if __name__ == "__main__":
    
    # Carrega a imagem, transforma em B&W e extrai os contornos 
    img = cv.imread("../data/img/art.png")
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # <200 -> dark | >200 -> white
    _, th = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)

    cv.imwrite("../data/img/art_th.jpg", th)

    # extrai contornos da imagem 
    contours, hierarchy = cv.findContours(
        th, 
        cv.RETR_EXTERNAL, 
        cv.CHAIN_APPROX_SIMPLE
    )

    drawn = cv.drawContours(img.copy(), contours, -1, (100,100,0), 10)

    cv.imwrite("../data/img/art_contour.jpg", drawn)

    art_contour = cv.imread("../data/img/art_contour.jpg")

    print(f"Encontrados {len(contours)} contornos na imagem.")

    # Geometria Analitica: x_m = sum_x / N, y_m = sum_y / N
    # Para cada contorno, calcular o ponto médio (Xmedio = sum_x/N, Ymedio = sum_y/N) e armazenar em uma lista
    medium_points_contours = []

    for contour in contours:
        sum_x = 0
        sum_y = 0
        n = len(contour)

        # Percorre cada ponto do contorno, aumentando a soma das coordenadas X e Y
        for point in contour:
            x, y = point[0]  
            sum_x += x
            sum_y += y

        # Calcula as médias (Xmedio e Ymedio)
        x_medio = int(sum_x / n)
        y_medio = int(sum_y / n)

        medium_points_contours.append((x_medio, y_medio))

    print(f"Contorno com {n} pontos -> Centro: ({x_medio}, {y_medio})")

    print(f'Centros dos contornos: {medium_points_contours}')

    # Desenha um circulo nos pontos médios dos contornos
    
    img_with_centers = art_contour.copy()

    for x_medio, y_medio in medium_points_contours:
        cv.circle(img_with_centers, (x_medio, y_medio), 5, (100, 100, 100), -1)

    cv.imwrite("../data/img/art_contour_with_centers.jpg", img_with_centers)

    # Comparar as cores dos pontos médios dos contornos para encontrar os mais semelhantes e os mais diferentes

    similar = defaultdict(set)
    similar_colors_used = set()
    different = defaultdict(set)

    for i, (x_medio, y_medio) in enumerate(medium_points_contours):
        for j, (x_medio2, y_medio2) in enumerate(medium_points_contours):
            if j > i:
                color1 = tuple(int(x) for x in art_contour[y_medio, x_medio])
                color2 = tuple(int(x) for x in art_contour[y_medio2, x_medio2])

                # diferença entre as cores usando a distância dos pixels
                dist = np.sqrt(
                    (int(color1[0]) - int(color2[0]))**2
                    + (int(color1[1]) - int(color2[1]))**2
                    + (int(color1[2]) - int(color2[2]))**2)
                
                print(f'Colors: {color1} vs {color2}')
                print(f'Distance: {dist}')

                # Considera os pontos medios com cores semelhantes 
                # se a distância entre as cores desses pontos for menor ou igual a 30
                if dist <= 30:
                    avg_color = (
                        (int(color1[0]) + int(color2[0])) // 2, 
                        (int(color1[1]) + int(color2[1])) // 2,
                        (int(color1[2]) + int(color2[2])) // 2
                    )
                    similar[avg_color].add((x_medio, y_medio))
                    similar[avg_color].add((x_medio2, y_medio2))
                    similar_colors_used.add(color1)
                    similar_colors_used.add(color2)
                else:
                    different[color1].add((x_medio, y_medio))
                    different[color2].add((x_medio2, y_medio2))

    # tirando as cores similares da lista de cores diferentes
    keys_to_remove = set()
    for sim_color in similar_colors_used:
        for diff_color in different:
            dist = np.sqrt(
                (sim_color[0] - diff_color[0])**2
                + (sim_color[1] - diff_color[1])**2
                + (sim_color[2] - diff_color[2])**2)
            if dist <= 30:
                keys_to_remove.add(diff_color)
    
    for key in keys_to_remove:
        different.pop(key, None)

    for s in similar:
        print(f'Similar color: {s} -> Points: {similar[s]}')
    
    for d in different:
        print(f'Different color: {d} -> Point: {different[d]}')