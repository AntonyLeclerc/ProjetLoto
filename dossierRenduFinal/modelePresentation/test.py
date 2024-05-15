import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Définition d'une fonction simple renvoyant une image
def openImage(f):
    return cv2.imread(f)

class LossAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))


# ===


model = tf.keras.models.load_model("./noyau10x10_10epochs_images70x70_couleur_flip_shift.h5")

# ===

x_train = []
y_train = []

for classe in range(1,51):
    for num in range(1,10):
        if os.path.isfile(f"../../ImagesRognees/num{classe}/{classe}_num_{num}.jpg"):
            img = openImage(f"../../ImagesRognees/num{classe}/{classe}_num_{num}.jpg")

            x_train.append(img)
            y_train.append(classe)

            
x_train = np.array(x_train)
y_train = np.array(y_train)
print(f"Here : {x_train[0].shape}")
# ===
print("Debut generation")

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=360,      # Rotate the image by a random angle within the specified range
    width_shift_range=0.1,  # Shift the image horizontally by a fraction of its width
    height_shift_range=0.1, # Shift the image vertically by a fraction of its height
    zoom_range=0.1,         # Zoom in or out on the image by a random factor
    horizontal_flip=False,   # Flip the image horizontally
    vertical_flip=False,    # Do not flip the image vertically
    fill_mode='nearest'     # Fill in missing pixels with the nearest value
)

# Reshape x_train to have 4 dimensions (assuming grayscale images)
x_train_augmented = np.reshape(x_train, (-1, 70, 70, 3))

# Initialize an empty array to store augmented images and labels
augmented_images = []
augmented_labels = []

# Generate augmented images
desired_augmentation_size = 5000  # Specify the desired size of the augmented dataset
batch_size = 32  # Specify a reasonable batch size
for x_batch, y_batch in datagen.flow(x_train_augmented, y_train, batch_size=batch_size):
    augmented_images.append(x_batch)
    augmented_labels.append(y_batch)
    if len(augmented_images) * batch_size >= desired_augmentation_size:
        break

# Stack augmented images and labels into single arrays
augmented_images = np.vstack(augmented_images)
augmented_labels = np.hstack(augmented_labels)

# Shuffle augmented data
shuffled_indices = np.random.permutation(len(augmented_images))
x_train_augmented = augmented_images[shuffled_indices]
y_train_augmented = augmented_labels[shuffled_indices]

print("Fin generation")

tmp = []
for img in x_train_augmented:
    tmpImg = img.astype(int)
    tmp.append(tmpImg)

x_train_augmented = np.array(tmp)

x_train_augmented = x_train_augmented / 255.0
x_train, x_test, y_train, y_test = train_test_split(x_train_augmented, y_train_augmented, test_size=0.2, random_state=42)

print(f"x_train : {x_train.shape}")
print(f"x_test : {x_test.shape}")

# ===


# ===
callback = LossAccuracyCallback()

for i in range(1,21):
    plt.subplot(4,5,i)
    c = np.random.randint(len(x_train))
    plt.imshow(x_train[c])
    plt.title(f"Classe : {y_train[c]}")

plt.show()






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