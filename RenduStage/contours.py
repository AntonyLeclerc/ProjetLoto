import os
import cv2 as cv
import numpy as np
import gc
from utils import *

"""
CE FICHIER LÀ A SURTOUT SERVI POUR CRÉER DES IMAGES POUR ILLUSTRER LES EXTRACIONS

"""

def merge_rectangles(rectangles, threshold=50):
    """ Fusionner les rectangles proches jusqu'à ce qu'il n'y ait plus de fusion possible """
    def get_center(rect):
        x, y, w, h = rect
        return (x + w // 2, y + h // 2)

    def get_union(rect1, rect2):
        (x1, y1, w1, h1) = rect1
        (x2, y2, w2, h2) = rect2
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    merged_rectangles = []
    lines = []
    used = [False] * len(rectangles)

    for i in range(len(rectangles)):
        if used[i]:
            continue
        current_rect = rectangles[i]
        union_rect = current_rect
        current_centers = [get_center(current_rect)]
        for j in range(i + 1, len(rectangles)):
            if used[j]:
                continue
            next_rect = rectangles[j]
            if are_boxes_close(*current_centers[0], *get_center(next_rect), threshold):
                next_center = get_center(next_rect)
                current_centers.append(next_center)

                union_rect = get_union(union_rect, next_rect)
                used[j] = True

        if len(current_centers) > 1:
            for k in range(len(current_centers) - 1):
                for l in range(k + 1, len(current_centers)):
                    lines.append((current_centers[k], current_centers[l]))

        merged_rectangles.append(union_rect)

    return merged_rectangles, lines

# Simulation des valeurs de classe et num
classe = np.random.randint(1, 50)

folder_path = filedialog.askdirectory()

# Pour les tests
for classe in range(1, 51):
    for ex in range(1, 6):
        num = np.random.randint(1, 200)
        gc.collect()
        # Charger les images
        base_path = f"{folder_path}/Boule{classe}/{classe}_num_{num}.jpg"


        img_colored = cv.imread(base_path)
        img_colored = cv.cvtColor(img_colored, cv.COLOR_BGR2RGB)
        img_colored_copy = img_colored.copy()

        img_rectangles_only = np.copy(img_colored)
        img_rectangles_line = np.copy(img_colored)
        img_contours_only = np.copy(img_colored)
        img_both = np.copy(img_colored)
        img_combined = np.copy(img_colored)  # Assurez-vous de copier img_colored pour img_combined

        img_gray = cv.cvtColor(img_colored, cv.COLOR_RGB2GRAY)

        # Créer une image binaire en changeant les pixels colorés en blanc
        img_bw = np.where(img_colored > 50, 255, 0).astype(np.uint8)
        img_bw = cv.cvtColor(img_bw, cv.COLOR_RGB2GRAY)

        # Appliquer un seuillage sur l'image en niveaux de gris
        _, thresh_gray = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY)

        # Détecter les contours sur l'image seuillée
        contours_gray, hierarchy_gray = cv.findContours(thresh_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Filtrer les contours par leur aire
        filtered_contours = [contour for contour in contours_gray if is_area_ok(contour, 1000, 30000)]

        # Créer des rectangles englobants autour des contours filtrés
        rectangles = [cv.boundingRect(contour) for contour in filtered_contours]

        # Filtrer les rectangles selon le ratio de pixels noirs
        filtered_rectangles = []
        for (x, y, w, h) in rectangles:
            if calculate_black_pixel_ratio(((x, y), (x + w, y + h)), img_bw) > 0.2:
                filtered_rectangles.append((x, y, w, h))

        # Fusionner les rectangles proches uniquement dans img_combined
        merged_rectangles, lines = merge_rectangles(filtered_rectangles, threshold=300)

        # Dessiner les rectangles fusionnés sur img_combined
        for (x, y, w, h) in merged_rectangles:
            cv.rectangle(img_combined, (x, y), (x + w, y + h), (255, 0, 255), 3)

        # Dessiner les contours filtrés et les rectangles non fusionnés sur les autres images
        cv.drawContours(img_contours_only, filtered_contours, -1, (0, 255, 0), 3)

        for (x, y, w, h) in filtered_rectangles:
            cv.rectangle(img_rectangles_only, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.rectangle(img_rectangles_line, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.rectangle(img_both, (x, y), (x + w, y + h), (255, 0, 255), 3)

        for (centerX1, centerY1), (centerX2, centerY2) in lines:
            cv.line(img_rectangles_line, (centerX1, centerY1), (centerX2, centerY2), (255, 0, 255), 5)
            
        cv.drawContours(img_both, filtered_contours, -1, (255, 0, 255), 3)

        # Sauvegarder les images avec OpenCV
        cv.imwrite(f"./contours/Boule{classe}/Ex{ex}/Rects.jpg", cv.cvtColor(img_rectangles_only, cv.COLOR_RGB2BGR))
        cv.imwrite(f"./contours/Boule{classe}/Ex{ex}/Cnts.jpg", cv.cvtColor(img_contours_only, cv.COLOR_RGB2BGR))
        cv.imwrite(f"./contours/Boule{classe}/Ex{ex}/Both.jpg", cv.cvtColor(img_both, cv.COLOR_RGB2BGR))
        cv.imwrite(f"./contours/Boule{classe}/Ex{ex}/RectLine.jpg", cv.cvtColor(img_rectangles_line, cv.COLOR_RGB2BGR))
        cv.imwrite(f"./contours/Boule{classe}/Ex{ex}/CombinedRects.jpg", cv.cvtColor(img_combined, cv.COLOR_RGB2BGR))
