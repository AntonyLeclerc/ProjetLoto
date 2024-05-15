import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model("./noyau5x5_20epochs/noyau5x5_20epochs.h5")

def openImage(f, gscale=False):
    
    if gscale:
        # Lecture de l'image en niveaux de gris
        img_gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        
        # Ajoute un canal pour les couleurs (ici en niveaux de gris)
        img = np.expand_dims(img_gray, axis=-1)
    else:
        img = cv2.imread(f)
    
    return img

class LossAccuracyCallback(tf.keras.callbacks.Callback):

    # Définit un objet qui récupèrera les accuracies et les loss lors de l'entraînement

    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Fonction appelée à la fin d'une période lors de l'entraînement
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))

def preprocess_image(image):
    # Convertit l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Application d'un flou Gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Application d'un seuillage adaptatif pour séparer l'arrière plan de l'avant plan (nos numéros)
    # Le seuillage adaptatif calcule un seuil pour des petites régions d'une image, au lieu d'un seuil unique pour l'image complète
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    

    # Remplit les éventuel "trous" causés par le seuillage dans les numéros, selon un noyau de taille 3x3
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    # Retransforme une image en noir et blanc en une image en couleurs, passée par la suite au réseau de neurones
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    return processed_image_rgb

x_train = []
y_train = []

print(f"Debut")
for classe in range(1,51):
    for num in range(1,10):
        if os.path.isfile(f"../ImagesRogneesTest/{classe}_num_{num}.jpg"):
            img = openImage(f"../ImagesRogneesTest/{classe}_num_{num}.jpg")

            # Si un pré-traitement est souhaité (défini au début), alors on pré-traite notre image
            img = preprocess_image(img)
            x_train.append(img)
            y_train.append(classe)
print(f"Fin")

x_train = np.array(x_train)
y_train = np.array(y_train)

print(f"Here : {x_train.shape}")
print(f"Début génération")
# Outil permettant l'augmentation artificielle de notre dataset
datagen = ImageDataGenerator(
    rotation_range=360,      # Rotation d'un angle aléatoire entre 0 et l'angle spéficié
    width_shift_range=0.1,  # Décalage horizontal de l'image jusqu'à 10% ici
    height_shift_range=0.1, # Décalage vertical de l'image jusqu'à 10% ici
    zoom_range=0.1,         # Zoom intérieur ou extérieur de l'image jusqu'à 10%
    horizontal_flip=False,   # Permet le retournement de l'image horizontal (on l'interdit ici)
    vertical_flip=False,    # Permet le retournement de l'image vertical (on l'interdit ici)
    fill_mode='nearest'     # Remplissage des pixels disparaissant (causés par les décalage / la rotation) par le pixel le plus proche 
)

x_train_augmented = x_train
# Listes récupérant les images générées, et leurs labels
augmented_images = []
augmented_labels = []

# Génération des images
desired_augmentation_size = 30000  # Nombre d'images générées souhaitées
batch_size = 32  # Nombre d'images générées à chaque itération
for x_batch, y_batch in datagen.flow(x_train_augmented, y_train, batch_size=batch_size):
    augmented_images.append(x_batch) # Ajout des images de notre nouveau batch
    augmented_labels.append(y_batch) # Ajout des labels des images de notre nouveau batch
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


# Normalisation de nos données
x_train_augmented = x_train_augmented / 255.0

# Séparation de nos données en un ensemble d'entraînement et un ensemble de test
x_train, x_test, y_train, y_test = train_test_split(x_train_augmented, y_train_augmented, test_size=0.2, random_state=42)


# 2 print simplement pour voir le nombre d'images par ensemble, le ratio doit être
print(f"x_train : {x_train.shape}")
print(f"x_test : {x_test.shape}")

model.evaluate(x_test, y_test)

taken = []
for _ in range(3):
    plt.figure(figsize=(12, 12))

    for i in range(1, 21):
        # Randomly select an index
        c = np.random.randint(0, len(x_test))
        if c in taken:
            while c in taken:
                c = np.random.randint(0, len(x_test))
        taken.append(c)
        
        # Extract the image
        img = x_test[c]
        
        # Reshape the image array to match the model's input shape
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Perform prediction
        prediction = model.predict(img)
        
        # Determine if the prediction is correct
        predicted_class = np.argmax(prediction)
        actual_class = y_test[c]
        is_correct = (predicted_class == actual_class)

        # Display the image and prediction
        plt.subplot(4, 5, i)
        plt.imshow(img.squeeze())  # Squeeze to remove batch dimension for display
        title = f"Class: {y_test[c]}\nPrediction: {predicted_class}"
        if is_correct:
            plt.title(title, color='green')
        else:
            plt.title(title, color='red')
        plt.axis("off")

    plt.show()

# ===