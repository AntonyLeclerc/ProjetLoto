import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# model = tf.keras.models.load_model("dossierRenduFinal/MP/noyau5x5_10epochs_images70x70_flip_shift_preprocessed.h5")

# Pour la présentation
model = tf.keras.models.load_model("./noyau5x5_20epochs/noyau5x5_20epochs.h5")


""" model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(70, 70, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(51, activation='softmax'))  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) """

class DragDropImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drag and Drop Image")
        self.canvas = tk.Canvas(self.root, width=400, height=450, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_drag_start)

        self.image = None
        self.numpyImage = None
        self.image_obj = None
        self.image_path = None
        self.text_label = None

        self.create_menu()

    def create_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file_dialog)

    def load_image(self, image_path):
        self.image_path = image_path
        image = Image.open(image_path)
        
        # Passe une image de dimension (70,70,4) si canal alpha en (70,70,3)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        self.numpyImage = np.array(image)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.delete(self.image_obj)  # Supprime l'ancienne image si elle existe
        self.image_obj = self.canvas.create_image(200, 200, image=self.image)

        # Ajout du label à l'importation
        classe, proba = predictImage(model, self.numpyImage)

        if self.text_label:  # Supprime l'ancien label si il existe
            self.text_label.destroy()

        self.text_label = tk.Label(self.canvas, text=f"Classe : {classe}\nProbabilité : {proba}", fg="black", font=("Helvetica", 12))
        self.canvas.create_window(200, 350, window=self.text_label)

    def on_drag_start(self, event):
        if not self.image_obj:
            self.open_file_dialog()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.load_image(file_path)
            self.canvas.unbind("<Button-1>")  # Disable further dragging


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

def predictImage(m, img):

    toPredict = preprocess_image(img)
    toPredict = toPredict / 255.0
    toPredict = np.expand_dims(toPredict, axis=0)  # Add batch dimension
        
    # Prédit l'image
    prediction = m.predict(toPredict)
    ind = np.argmax(prediction)

    return ind, prediction[0][ind]


if __name__ == "__main__":
    root = tk.Tk()
    app = DragDropImageApp(root)
    root.mainloop()
