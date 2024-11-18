import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import getModel
from parseImages import parseImages
from sklearn.preprocessing import LabelEncoder

[images, labels] = parseImages('road-sign-detection/annotations', 'road-sign-detection/images')

images = images / 255
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = getModel(len(np.unique(labels)))

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Dokładność na zbiorze testowym: {test_accuracy * 100:.2f}%")

# Wykres dokładności i straty
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treningowa')
plt.plot(history.history['val_accuracy'], label='Walidacyjna')
plt.title('Dokładność')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treningowa')
plt.plot(history.history['val_loss'], label='Walidacyjna')
plt.title('Strata')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.show()