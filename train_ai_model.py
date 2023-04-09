import os
import cv2
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier

samples = np.empty((0, 100))
responses = []

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
    return images

def process_images(images, label):
    global samples, responses

    for img in images:
        img_resized = cv2.resize(img, (10, 10))
        responses.append(label)
        sample = img_resized.reshape((1, 100))
        samples = np.append(samples, sample, 0)

# Load images and process them
open_garage_images = load_images_from_folder("snapshots/open")
closed_garage_images = load_images_from_folder("snapshots/closed")

process_images(open_garage_images, 1)  # Label 1 for open garage doors
process_images(closed_garage_images, 0)  # Label 0 for closed garage doors

# Train the k-NN model
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(samples, responses.ravel())

# Save the trained model
np.savetxt('generalsamples.data', samples)
np.savetxt('generalresponses.data', responses)
