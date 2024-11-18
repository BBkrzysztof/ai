import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_model
from sklearn.preprocessing import LabelEncoder
from loaddataset import *

IMG_SIZE = 64
NUM_CLASSES = 15  # Liczba klas w Twoim zbiorze danych

train_images_dir = 'road-sign-detection/train/images'
train_labels_dir = 'road-sign-detection/train/labels/'

# Wczytanie danych treningowych
train_data = load_dataset(train_images_dir, train_labels_dir, IMG_SIZE)

# # Podział na obrazy i etykiety
train_images = np.array([item[0] for item in train_data])
train_labels = np.array([item[1] for item in train_data])

# # Przygotowanie etykiet w formacie one-hot
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)

print(train_images.shape)
print(train_labels.shape)


input_shape = (IMG_SIZE, IMG_SIZE, 3)
model = create_model(input_shape, NUM_CLASSES)

# # # Trenowanie modelu
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# # # Zapisanie modelu
model.save('yolo_model.h5')

