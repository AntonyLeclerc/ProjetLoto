import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tkinter import filedialog

def are_boxes_close(centerX1, centerY1, centerX2, centerY2, threshold):
    """
    Permet de dire si deux zones détectées sont suffisamment proches
    Sera utilisée lors de la fusion de zones (pour des nombres à 2 chiffres)
    """
    return np.sqrt((centerX2 - centerX1)**2 + (centerY2 - centerY1)**2) < threshold

def calculate_black_pixel_ratio(rect, img_gray2):
    """
    Permet d'enlever une zone si celle ci contient trop de pixels blancs sur une image seuillée
    On fait l'hypothèse ici qu'une image ayant trop de pixels blanc ne sera pas une zone intéressante (pas un chiffre / nombre) 
    """
    (x1, y1), (x2, y2) = rect
    roi = img_gray2[y1:y2, x1:x2]
    black_pixels = np.sum(roi == 0)
    total_pixels = roi.size
    return black_pixels / total_pixels if total_pixels > 0 else 0

def is_area_ok(contour, minArea, maxArea):
    """
    Permet de dire si une zone détectée n'est ni trop petite, ni trop grande
    """
    area = cv.contourArea(contour)
    return minArea <= area <= maxArea

def import_images(folder, p):
    X = []
    Y = []
    for classe in range(1,51):
        for num in range(1,900):
            if np.random.random() < p and os.path.isfile(f"{folder}/Boule{classe}/{classe}_num_{num}.jpg"):
                img = cv.imread(f"{folder}/Boule{classe}/{classe}_num_{num}.jpg")
                X.append(img)
                Y.append(classe)

    return X,Y

def transform_img(img, grayscale=False, otsu=False, preprocessing=False):
    """
    Transforme une image selon une modification voulue
    """
    tmp=None

    if grayscale:

        tmp = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        tmp = np.expand_dims(tmp, axis=-1)
        return tmp

    elif otsu:

        tmp = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(tmp,(5,5),0)
        ret3,tmp = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        tmp = np.expand_dims(tmp, axis=-1)
        return tmp

    elif preprocessing:

        tmp = preprocess_image(img)
        tmp = np.expand_dims(tmp, axis=-1)
        return tmp

    else:
        return img

def normalize_images(images):
    """
    Normalisation des valeurs de nos images
    """
    min_val = np.min(images)
    max_val = np.max(images)
    images = (images - min_val) / (max_val - min_val)
    return images

# Définition d'une fonction simple renvoyant une image
def openImage(f, gscale=False):
    if gscale:
        # Lecture de l'image en niveaux de gris
        img_gray = cv.imread(f, cv.IMREAD_GRAYSCALE)
        # Ajoute un canal pour les couleurs (ici en niveaux de gris)
        img = np.expand_dims(img_gray, axis=-1)
    else:
        img = cv.imread(f)
    return img
        

# Fonction appliquant un pré-traitement à nos images        
def preprocess_image(image):
    # Convertit l'image en niveaux de gris
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Application d'un flou Gaussien pour réduire le bruit
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # Application d'un seuillage adaptatif pour séparer l'arrière plan de l'avant plan (nos numéros)
    # Le seuillage adaptatif calcule un seuil pour des petites régions d'une image, au lieu d'un seuil unique pour l'image complète
    thresholded = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 4)
    # Remplit les éventuel "trous" causés par le seuillage dans les numéros, selon un noyau de taille 3x3
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)
    # Retransforme une image en noir et blanc en une image en couleurs, passée par la suite au réseau de neurones
    return processed_image
    
    
# Fonction permettant de générer des images à partir de nos images déjà existantes (dans les paramètres x, y)
def generate_images(nb_images, x, y, augmentation_artificielle_dataset=False, proportion=0.2):
    if augmentation_artificielle_dataset:
        print(f"Début génération")
        # Outil permettant l'augmentation artificielle de notre dataset
        datagen = ImageDataGenerator(
            rotation_range=20,      # Rotation d'un angle aléatoire entre 0 et l'angle spécifié
            width_shift_range=0.1,   # Décalage horizontal de l'image jusqu'à 10% ici
            height_shift_range=0.1,  # Décalage vertical de l'image jusqu'à 10% ici
            zoom_range=0.1,          # Zoom intérieur ou extérieur de l'image jusqu'à 10%
            horizontal_flip=False,   # Permet le retournement de l'image horizontal (on l'interdit ici)
            vertical_flip=False,     # Permet le retournement de l'image vertical (on l'interdit ici)
            fill_mode='nearest'      # Remplissage des pixels disparaissant (causés par les décalages / la rotation) par le pixel le plus proche 
        )

        # Copie des données d'entraînement
        x_train_augmented = x.copy()
        y_train_augmented = y.copy()

        # Listes récupérant les images générées, et leurs labels
        augmented_images = []
        augmented_labels = []

        # Génération des images
        desired_augmentation_size = nb_images  # Nombre d'images générées souhaitées
        batch_size = 32  # Nombre d'images générées à chaque itération
        for x_batch, y_batch in datagen.flow(x_train_augmented, y_train_augmented, batch_size=batch_size):
            augmented_images.append(x_batch)  # Ajout des images de notre nouveau batch
            augmented_labels.append(y_batch)  # Ajout des labels des images de notre nouveau batch
            if len(augmented_images) * batch_size >= desired_augmentation_size:
                break

        # On concatène toutes nos images et nos labels ensemble
        augmented_images = np.vstack(augmented_images)
        augmented_labels = np.hstack(augmented_labels)

        # Mélange de nos images et de nos labels selon une même permutation
        shuffled_indices = np.random.permutation(len(augmented_images))
        x_train_augmented = augmented_images[shuffled_indices]
        y_train_augmented = augmented_labels[shuffled_indices]

        print(f"Fin génération")
    else:
        x_train_augmented = x
        y_train_augmented = y

    # Normalisation de nos données
    x_train_augmented = normalize_images(x_train_augmented)

    # Séparation de nos données en un ensemble d'entraînement et un ensemble de test
    x_train, x_test, y_train, y_test = train_test_split(x_train_augmented, y_train_augmented, test_size=proportion, random_state=42)

    return x_train, x_test, y_train, y_test

def is_area_ok(contour, minArea, maxArea):
    area = cv.contourArea(contour)
    return minArea <= area <= maxArea

def get_regions(imageRGB):
    img_gray = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
    img_gray2 = np.where(img_gray < 80, 0, 255).astype(np.uint8)
    ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if is_area_ok(contour, 300, 20000)]


    allRect = []
    for contour in filtered_contours:
        x, y, w, h = cv.boundingRect(contour)
        allRect.append(((x, y), (x + w, y + h)))

    # Fusion des rectangles proches
    newRects = []
    processed = [False] * len(allRect)

    for i in range(len(allRect)):
        if processed[i]:
            continue
        (x1, y1), (x2, y2) = allRect[i]
        topLeft = (x1, y1)
        bottomRight = (x2, y2)

        for j in range(i + 1, len(allRect)):
            if processed[j]:
                continue
            (x3, y3), (x4, y4) = allRect[j]
            centerX1, centerY1 = (x1 + x2) // 2, (y1 + y2) // 2
            centerX2, centerY2 = (x3 + x4) // 2, (y3 + y4) // 2

            if are_boxes_close(centerX1, centerY1, centerX2, centerY2, 100):
                topLeft = (min(topLeft[0], x3), min(topLeft[1], y3))
                bottomRight = (max(bottomRight[0], x4), max(bottomRight[1], y4))
                processed[j] = True

        newRects.append((topLeft, bottomRight))

    # Filtrage des rectangles basés sur la proportion de pixels noirs
    filteredRects = []
    for rect in newRects:
        if calculate_black_pixel_ratio(rect, img_gray2) >= 0.20:
            filteredRects.append(rect)

    
    return filteredRects