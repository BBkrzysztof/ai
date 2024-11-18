import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from loaddataset import *


IMG_SIZE = 64
CLASS_TO_NAME = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']


# Wczytaj model
model = tf.keras.models.load_model('yolo_model.h5')

# Przykładowa predykcja
test_image, _ = load_image_and_labels(
    '50.jpg',
    'road-sign-detection/valid/labels/00000_00000_00002_png.rf.109f031ac8e60eba952da43b054389c0.txt',
      IMG_SIZE
    )

prediction = model.predict(np.expand_dims(test_image, axis=0))

# Wyświetl wynik
predicted_class = np.argmax(prediction)
print(f'Predicted class: {CLASS_TO_NAME[predicted_class]}')
plt.imshow(test_image)
plt.show()
