import argparse
import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

video_stream = 'rtsp://localhost:51610/a5793736882c5dbc'
mqtt_endpoint = '10.0.1.204'
mqtt_queue = '/garage/door'
mqtt_username = 'mqtt'
mqtt_password = 'mqtt'

cap = cv2.VideoCapture(video_stream)

def reconnect_stream():
    global cap
    cap.release()
    cap = cv2.VideoCapture(video_stream)

def on_connect(client, userdata, flags, rc):
    print("Connected! Result code: " + str(rc))

print("Connecting to MQTT...")
mqtt_client = mqtt.Client("garage_door")
mqtt_client.username_pw_set(mqtt_username, mqtt_password) 
mqtt_client.on_connect = on_connect
mqtt_client.loop_start()
mqtt_client.connect(mqtt_endpoint)
while not mqtt_client.is_connected():
    time.sleep(1)

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(samples, responses.ravel())

def process_image(img):
    x1, y1, x2, y2 = 268, 1, 542, 112
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

import argparse
import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

video_stream = 'rtsp://localhost:51610/a5793736882c5dbc'
mqtt_endpoint = '10.0.1.204'
mqtt_queue = '/garage/door'
mqtt_username = 'mqtt'
mqtt_password = 'mqtt'

cap = cv2.VideoCapture(video_stream)

def reconnect_stream():
    global cap
    cap.release()
    cap = cv2.VideoCapture(video_stream)

def on_connect(client, userdata, flags, rc):
    print("Connected! Result code: " + str(rc))

print("Connecting to MQTT...")
mqtt_client = mqtt.Client("garage_door")
mqtt_client.username_pw_set(mqtt_username, mqtt_password) 
mqtt_client.on_connect = on_connect
mqtt_client.loop_start()
mqtt_client.connect(mqtt_endpoint)
while not mqtt_client.is_connected():
    time.sleep(1)

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(samples, responses.ravel())

def process_image(img):
    x1, y1, x2, y2 = 268, 1, 542, 112
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

def get_garage_door_state():
    ret, img = cap.read()
    if not ret:
        reconnect_stream()
        return get_garage_door_state()

    img_cropped = process_image(img)
    cv2.imshow("Cropped Image", img_cropped)  # Display the cropped image
    cv2.waitKey(1)

    img_flat = img_cropped.reshape((1, -1))  # Flatten the cropped image
    img_float = np.float32(img_flat)

    prediction = knn.predict(img_float)
    prediction_probabilities = knn.predict_proba(img_float)
    print("Prediction probabilities:", prediction_probabilities)  # Print the classifier's predicted probabilities

    return "open" if prediction[0] == 1 else "closed"

while True:
    garage_door_state = get_garage_door_state()
    print("Garage door state:", garage_door_state)

    mqtt_client.publish(mqtt_queue, garage_door_state)
    time.sleep(5)

cv2.destroyAllWindows()  # Close the OpenCV windows when the script ends


while True:
    garage_door_state = get_garage_door_state()
    print("Garage door state:", garage_door_state)

    mqtt_client.publish(mqtt_queue, garage_door_state)
    time.sleep(5)

cv2.destroyAllWindows()  # Close the OpenCV windows when the script ends
