import tensorflow as tf
import numpy as np
import os
import cv2

def load_image_and_labels(image_path, label_path, img_size):
    # Wczytaj obraz
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0  # Normalizacja

    # Wczytaj etykiety
    class_id = 0
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_id = int(class_id)
            break

    return image, class_id

def load_dataset(images_dir, labels_dir, img_size):
    images = sorted(os.listdir(images_dir))
    labels = sorted(os.listdir(labels_dir))
    dataset = []

    for img_file, label_file in zip(images, labels):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        image, label = load_image_and_labels(img_path, label_path, img_size)
    
        dataset.append((image, label))

    return dataset
