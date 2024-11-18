import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def parseImages(annotations_dir,images_dir):

    images =[]
    labels = []

    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        file_name = root.find('filename').text
        image_path = os.path.join(images_dir, file_name)
        image = cv2.imread(image_path)

        for obj in root.findall('object'):
            label = obj.find('name').text

            bbox = obj.find('bndbox')

            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            cropped_image = image[ymin:ymax, xmin:xmax]
            resized_image = cv2.resize(cropped_image,(64,64))
            labels.append(label)
            images.append(resized_image)
    
    return [np.array(images),np.array(labels)]