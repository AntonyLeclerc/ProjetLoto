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
nbImagesParClasse = 300


# A mettre sur True si l'on souhaite travailler avec des images en niveaux de gris
grayscale = input(f"Passer vos images en niveaux de gris ?\nO : Oui\nN : Non\nVotre choix : ")
while grayscale != "O" and grayscale != "N":
	grayscale = input(f"Votre choix : ")
grayscale = True if grayscale == "O" else False

# Méthode "Otsu" de seuillage

if not grayscale:
	otsu = input(f"Appliquer un seuillage 'Otsu' à vos images ?\nO : Oui\nN : Non\nVotre choix : ")
	while otsu != "O" and otsu != "N":
		otsu = input(f"Votre choix : ")
	otsu = True if otsu == "O" else False

# A mettre sur True si l'on souhaite pré-traiter les images

if not(grayscale or otsu):
	preprocessing = input(f"Appliquer un pré-traitement à vos images ?\nO : Oui\nN : Non\nVotre choix : ")
	while preprocessing != "O" and preprocessing != "N":
		preprocessing = input(f"Votre choix : ")
	preprocessing = True if preprocessing == "O" else False
	
	
if grayscale or otsu or preprocessing:
	modified = True
else:
	modified = False


# A mettre sur True si l'on souhaite augmenter artificiellement notre dataset
# Préférer le mettre à True si notre dataset contient peu d'éléments
augmentation_artificielle_dataset = input(f"Augmenter artificiellement votre dataset (des modifications légères seront apportées aux images) ?\nO : Oui\nN : Non\nVotre choix : ")
while augmentation_artificielle_dataset != "O" and augmentation_artificielle_dataset != "N":
	augmentation_artificielle_dataset = input(f"Votre choix : ")
augmentation_artificielle_dataset = True if augmentation_artificielle_dataset == "O" else False

# Le nombre d'images souhaitées, dans le cas où l'on choisit d'augmenter notre dataset
if augmentation_artificielle_dataset:
    nombre_dimages_souhaitees = input("Combien d'images souhaitez vous générer : ")
    while not(nombre_dimages_souhaitees.isdigit()):
        nombre_dimages_souhaitees = input("Réessayez : ")

    nombre_dimages_souhaitees = int(nombre_dimages_souhaitees)



# Chemin jusqu'aux images
print(f"Veuillez sélectionner le dossier ou vos images sont contenues")
folder_path = filedialog.askdirectory()

nbEpochs = input(f"Nombre de périodes durant lequel le réseau devra s'entraîner :")
while not(nbEpochs.isdigit()):
	nbEpochs = input(f"Réessayez : ")

nbEpochs = int(nbEpochs)

# Noyau de convolution

kernelSize = input(f"Taille du noyau de convolution (entrez uniquement un nombre) :")
while not(kernelSize.isdigit()):
	kernelSize = input(f"Réessayez : ")

kernelSize = int(kernelSize)

lr = 1e-3

# Permet de récupérer la loss et l'accuracy automatiquement à chaque epoch
class LossAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Fonction appelée à la fin d'une période lors de l'entraînement
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))


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



X = []
Y = []


# Importation des images et extraction des zones
for classe in range(1, 51):
    taken = []
    print(f"Classe = {classe}")
    for _ in range(1, nbImagesParClasse):

        num = np.random.randint(1, 500)
        while num in taken:
            num = np.random.randint(1, 500)

        taken.append(num)
        # Libération de la mémoire déréférencée
        gc.collect()
        # Charger les images
        base_path = f"{folder_path}/Boule{classe}/{classe}_num_{num}.jpg"

        img_colored = cv.imread(base_path)
        img_extraction = np.copy(img_colored)
        #def transform_img(img, grayscale=False, otsu=False, preprocessing=False):
        img_extraction = transform_img(img_extraction, grayscale, otsu, preprocessing)

        img_colored = cv.cvtColor(img_colored, cv.COLOR_BGR2RGB)
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


        # Une fois toutes les régions détectées 

        for (x,y,w,h) in merged_rectangles:
            x1,y1, x2,y2 = x, y, x+w, y+h
            centerX, centerY = (x1+x2)//2, (y1+y2)//2

            toExtract = img_extraction[centerY-120:centerY+120, centerX-120:centerX+120]
            n,m,k = toExtract.shape
            if n == 240 and m == 240:
                toExtract = cv2.resize(toExtract, (100, 100), interpolation = cv2.INTER_CUBIC)
                X.append(toExtract)
                Y.append(classe)

        # Dessiner les contours filtrés et les rectangles non fusionnés sur les autres images
        #for (x, y, w, h) in merged_rectangles:
        #    cv.rectangle(img_combined, (x, y), (x + w, y + h), (255, 0, 255), 3)


X = np.array(X)
Y = np.array(Y)

X = np.expand_dims(X, axis=-1)
print(f"{X.shape=}")

shp = X[0].shape

# Création du réseau de neurones
inputs = tf.keras.Input(shape=shp)
x = tf.keras.layers.Conv2D(32, (kernelSize, kernelSize), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (kernelSize, kernelSize), activation='relu')(x)
x = tf.keras.layers.Conv2D(64, (kernelSize, kernelSize), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (kernelSize, kernelSize), activation='relu')(x)
x = tf.keras.layers.Conv2D(64, (kernelSize, kernelSize), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Transformation des nos images en vecteurs unidimensionnels
x = tf.keras.layers.Flatten()(x)

# Ajout de couches 'Dense' 
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

# Création d'une couche 'logits'
logits = tf.keras.layers.Dense(51)(x)

# Ajout de la dernière couche softmax (celle qui sera chargée de renvoyer un vecteur de probabilités)
output = tf.keras.layers.Activation('softmax')(logits)

# Creation du modèle final
final_model = tf.keras.models.Model(inputs=inputs, outputs=output)

"""
Compilation du modèle
Choix d'un optimiseur, d'une fonction de calcul de la loss, et d'une métrique (ici, seule l'accuracy nous intéresse)
"""
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


# Création du chemin de sauvegarde selon plusieurs paramètres pré-choisis
sp_cpt = 1
savePath = f"./noyau{kernelSize}x{kernelSize}_{nbEpochs}epochs"

if grayscale: savePath += "_grayscale"
if otsu: savePath += "_otsu"
if preprocessing: savePath += "_preprocessing"
if augmentation_artificielle_dataset: savePath += f"_augmented_{nombre_dimages_souhaitees}"

while os.path.exists(f"./{savePath}_{sp_cpt}"):
	sp_cpt += 1


savePath += f"_{sp_cpt}"
os.mkdir(f"./{savePath}")	

# Création de notre objet qui récupèrera nos loss et nos accuracy au cours des epochs
callback = LossAccuracyCallback()

proportion = 0.2

# Augmentation artificielle de notre jeu de donnée si voulu puis entraînement
if augmentation_artificielle_dataset:
    for i in range(1, nbEpochs+1):
        gc.collect()
        print(f"Epoch n°{i}")
        x_train, x_test, y_train, y_test = generate_images(nombre_dimages_souhaitees, X, Y, True, proportion)
        final_model.fit(x_train, y_train, epochs=1, callbacks=[callback], validation_data=(x_test, y_test))
        del x_train
        del x_test
        del y_train
        del y_test

    x_train, x_test, y_train, y_test = generate_images(nombre_dimages_souhaitees, X, Y, True, proportion)
    final_model.evaluate(x_test, y_test)
    del x_train
    del y_train
    gc.collect()

else:
    """
    Si pas d'augmentation artificielle, sépération de notre jeu importé en un ensemble d'entraînement et de test
    Puis entraînement
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=proportion, random_state=42)
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    final_model.fit(x_train, y_train, epochs=nbEpochs, callbacks=[callback], validation_data=(x_test, y_test))

    final_model.evaluate(x_test, y_test)

# Sauvegarde de notre modèle et de ses paramètres
final_model.save(f"./{savePath}/model.h5")

# Création de nos images de tests visuels
if True:
    cpt = 1
    taken = []
    for _ in range(5):
        plt.figure(figsize=(12, 16))
        for i in range(1, 21):
            # Randomly select an index
            c = np.random.randint(0, len(x_test))
            taken.append(c)
            
            # Extract the image
            img = x_test[c]
            
            # Reshape the image array to match the model's input shape
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Perform prediction
            prediction = final_model.predict(img)
            
            # Determine if the prediction is correct
            predicted_class = np.argmax(prediction)
            actual_class = y_test[c]
            is_correct = (predicted_class == actual_class)

            # Display the image and prediction
            plt.subplot(4, 5, i)
            if modified:
                plt.imshow(img.squeeze(), cmap="gray")  # Squeeze to remove batch dimension for display
            else:
            	plt.imshow(img.squeeze())  # Squeeze to remove batch dimension for display
            title = f"Class: {y_test[c]}\nPrediction: {predicted_class}\nProba : {prediction[0][predicted_class]:.3f}"
            if is_correct:
                plt.title(title, color='green')
            else:
                plt.title(title, color='red')
            plt.axis("off")
        
        plt.savefig(f"./{savePath}/visualTest_{cpt}.png")
        cpt += 1

loss = callback.losses
acc = callback.accuracies

plt.figure(figsize=(12, 12))
# Affichage des courbes de loss et d'accuracy
plt.plot(range(1, nbEpochs+1), loss, color="red", label="loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.savefig(f"./{savePath}/loss.png")

plt.figure(figsize=(12, 12))
plt.plot(range(1, nbEpochs+1), acc, color="blue", label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig(f"./{savePath}/accuracy.png")

