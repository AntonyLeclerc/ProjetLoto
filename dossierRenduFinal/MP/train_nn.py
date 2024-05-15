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
    # Read the image in grayscale mode
    img = cv2.imread(f)
    
    return img

class LossAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to separate foreground from background
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    # Morphological operations to further clean up the image
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    # Replicate single-channel image across all three channels
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    return processed_image_rgb

# ===

model = tf.keras.models.Sequential()

# Add convolutional layers
model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(70, 70, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Flatten layer to transition from convolutional layers to fully connected layers
model.add(tf.keras.layers.Flatten())

# Add dense (fully connected) layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(51, activation='softmax'))  # Assuming 50 classes (numbers 1-50)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===

x_train = []
y_train = []

for classe in range(1,51):
    for num in range(1,10):
        if os.path.isfile(f"../../ImagesRognees/num{classe}/{classe}_num_{num}.jpg"):
            img = openImage(f"../../ImagesRognees/num{classe}/{classe}_num_{num}.jpg")
            preprocessed = preprocess_image(img)
            x_train.append(preprocessed)
            y_train.append(classe)

            
x_train = np.array(x_train)
y_train = np.array(y_train)

# ===

print(f"Here : {x_train.shape}")

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
x_train_augmented = x_train


# Initialize an empty array to store augmented images and labels
augmented_images = []
augmented_labels = []

# Generate augmented images
desired_augmentation_size = 200000  # Specify the desired size of the augmented dataset
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

plt.figure(figsize=(12,20))
for i in range(1,21):
    plt.subplot(4,5,i)
    c = np.random.randint(len(x_train))
    plt.imshow(x_train[c])
    plt.title(f"Classe : {y_train[c]}")
plt.show()

print(f"Debut entraînement")
model.fit(x_train, y_train, epochs=10, callbacks=[callback])


#model.save("noyau5x5_10epochs_images70x70_flip_shift_preprocessed.h5")
#model = tf.keras.models.load_model("noyau10x10_10epochs_images70x70_flip_shift_preprocessed.h5")


model.evaluate(x_test, y_test)



# Use boolean indexing to extract images with the target label

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

loss = callback.losses
acc = callback.accuracies


plt.plot(range(1,11), loss, color="red", label="loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.show()

plt.plot(range(1,11), acc, color="blue", label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

