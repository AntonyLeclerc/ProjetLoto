import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def openImage(f, gscale=False):
    
    img = cv2.imread(f)
    
    return img


x_train = []
y_train = []

print(f"Debut")
for classe in range(1,51):
    for num in range(1,700):
        if os.path.isfile(f"./TargetSet2/num{classe}/{num}.jpg"):
            img = openImage(f"./TargetSet2/num{classe}/{num}.jpg")

            # Si un pré-traitement est souhaité (défini au début), alors on pré-traite notre image
            x_train.append(img)
            y_train.append(classe)

print(f"Fin")

plt.figure(figsize=(12,20))

taken = []

length = len(x_train)
print(f"Il y'a {length} images !")
for i in range(1,21):
    plt.subplot(4,5,i)
    c = np.random.randint(length)
    if c in taken:
        while c in taken:
            c = np.random.randint(length)

    plt.imshow(x_train[c])
    plt.title(f"Classe : {y_train[c]}")
    plt.axis("off")
plt.show()