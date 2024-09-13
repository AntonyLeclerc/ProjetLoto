import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import *
from tkinter import filedialog
import gc


grayscale = False
otsu = False
preprocessing = False

print(f"Indiquez où se trouve votre model")
folder_path = filedialog.askdirectory()

if "grayscale" in folder_path:
    grayscale = True

if "otsu" in folder_path:
    otsu = True

if "preprocessing" in folder_path:
    preprocessing = True

if grayscale or otsu or preprocessing:
    modified = True

model = tf.keras.models.load_model(f"{folder_path}/model.h5")

stop = False
cpt = 1

import cv2 as cv
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt

stop = False
cpt = 0

while not stop:
    allPreds = []
    print(f"Indiquez l'image à charger")
    img_path = filedialog.askopenfilename()

    img = cv.imread(f"{img_path}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_extraction = transform_img(img, grayscale=grayscale, otsu=otsu, preprocessing=preprocessing)
    rectangles = get_regions(img)

    regions = []
    for rect in rectangles:
        (x1,y1), (x2,y2) = rect
        centerX, centerY = (x1+x2)//2, (y1+y2)//2

        area = img_extraction[centerY-130:centerY+130, centerX-130:centerX+130]

        shp = area.shape

        if shp[0] == 260 and shp[1] == 260:
            area = cv.resize(area, (100, 100), interpolation=cv.INTER_CUBIC)
            regions.append(area)

    print(f"{len(regions)} régions ont été extraites")
    if len(regions) == 0:
        continue

    plt.figure(figsize=(12, 6))
    regions = normalize_images(regions)

    for i in range(len(regions)):
        tmp = regions[i]

        toPredict = np.expand_dims(tmp, axis=0)

        prediction = model.predict(toPredict)

        allPreds.append(prediction[0])

        argumax = np.argmax(prediction)

        proba = prediction[0][argumax]

        plt.subplot(4, 5, i+2)
        if modified:
            plt.imshow(tmp, cmap='gray')
        else:
            plt.imshow(tmp)
        plt.title(f"Pred : {argumax}\n{proba}")
        plt.axis("off")

    plt.subplot(4, 5, 1)
    plt.imshow(img)
    plt.title(f"Originale")
    plt.axis("off")

    # Ajouter du padding entre les sous-graphiques
    plt.subplots_adjust(hspace=0.5, wspace=0.5)


    allPreds = np.array(allPreds)
    preds_prod = np.prod(allPreds, axis=0)
    preds_mean = np.mean(allPreds, axis=0)

    plt.subplot(4,5,16)
    plt.text(0.5, 0.5, f'Produit de probabilités\nClasse : {np.argmax(preds_prod)}\nProba : {preds_prod[np.argmax(preds_prod)]}', fontsize=12, ha='center', va='center')
    plt.axis("off")

    plt.subplot(4,5,20)
    plt.text(0.5, 0.5, f'Moyenne de probabilités\nClasse : {np.argmax(preds_mean)}\nProba : {preds_mean[np.argmax(preds_mean)]}', fontsize=12, ha='center', va='center')
    plt.axis("off")
    
    plt.savefig(f"Preds_{cpt}")
    
    plt.show()

    choix = int(input(f"Voulez vous quitter ? 1 : Oui / 2 : Non\n"))

    if choix == 1:
        stop = True

    cpt += 1

