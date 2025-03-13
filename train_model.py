import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

# Paths
DATA_DIR_TRAIN = r"C:\internshhip\emotion detection and sentimental analysis\train"
DATA_DIR_TEST = r"C:\internshhip\emotion detection and sentimental analysis\test"
MODEL_PATH = "emotion_detection_model.h5"

IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 32
EPOCHS = 25

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR_TRAIN,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    DATA_DIR_TEST,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class_labels = list(train_generator.class_indices.keys())

# Load or Train Model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(len(class_labels), activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)
    model.save(MODEL_PATH)

def analyze_emotions(test_generator, model):
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    emotion_counts = {label: 0 for label in class_labels}
    for pred in predicted_classes:
        emotion_counts[class_labels[pred]] += 1
    
    total_images = len(predicted_classes)
    sentiment_percentages = {label: (count / total_images) * 100 for label, count in emotion_counts.items()}
    
    return sentiment_percentages

sentiment_result = analyze_emotions(test_generator, model)

def analyze_custom_images(image_paths, model):
    emotion_counts = {label: 0 for label in class_labels}
    total_images = len(image_paths)
    
    for img_path in image_paths:
        img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        emotion_counts[class_labels[predicted_class]] += 1
    
    sentiment_percentages = {label: (count / total_images) * 100 for label, count in emotion_counts.items()}
    return sentiment_percentages

def open_file_dialog():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_paths:
        result = analyze_custom_images(file_paths, model)
        messagebox.showinfo("Analysis Result", f"Custom Image Sentiment Analysis: {result}")

def drag_and_drop(event):
    file_paths = event.data.strip().split()
    result = analyze_custom_images(file_paths, model)
    messagebox.showinfo("Analysis Result", f"Custom Image Sentiment Analysis: {result}")

# GUI Setup
root = TkinterDnD.Tk()
root.title("Emotion Detection App")
root.geometry("400x300")

label = tk.Label(root, text=f"Model Accuracy: {sentiment_result}", font=("Arial", 12))
label.pack(pady=10)

analyze_button = tk.Button(root, text="Analyze Your Own Set of Images", command=open_file_dialog)
analyze_button.pack(pady=10)

drop_label = tk.Label(root, text="Or Drag & Drop Images Here", font=("Arial", 10))
drop_label.pack(pady=10)

drop_area = tk.Label(root, text="Drop Images Here", bg="lightgray", width=40, height=5)
drop_area.pack(pady=10)

drop_area.drop_target_register(DND_FILES)
drop_area.dnd_bind("<<Drop>>", drag_and_drop)

root.mainloop()
