import argparse
import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import threading

video_stream = 'rtsp://localhost:51610/a5793736882c5dbc'
mqtt_endpoint = '10.0.1.204'
mqtt_queue = '/garage/door'
mqtt_username = 'mqtt'
mqtt_password = 'mqtt'

cap = cv2.VideoCapture(video_stream)
current_frame = None
frame_lock = threading.Lock()

def reconnect_stream():
    global cap
    cap.release()
    cap = cv2.VideoCapture(video_stream)

def update_frame():
    global cap, current_frame, frame_lock
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                reconnect_stream()
            else:
                with frame_lock:
                    current_frame = frame
        except cv2.error as e:
            print(f"OpenCV error: {e}")
            reconnect_stream()
        time.sleep(0.1)

frame_update_thread = threading.Thread(target=update_frame)
frame_update_thread.daemon = True
frame_update_thread.start()


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

def clear_buffer():
    for _ in range(5):
        cap.grab()


def get_garage_door_state(img, retries=3):
    if retries == 0:
        return None

    img = process_image(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    img_resized = cv2.resize(img_gray, (10, 10))  # Resize the grayscale image
    img_reshaped = img_resized.reshape((1, 100))
    img_float = np.float32(img_reshaped)

    cv2.imshow("Input Image", img)  # Display the input image
    cv2.imshow("Grayscale Resized Image", img_resized)  # Display the grayscale resized image
    cv2.waitKey(1)

    prediction = knn.predict(img_float)
    prediction_probabilities = knn.predict_proba(img_float)
    print("Prediction probabilities:", prediction_probabilities)
    return "open" if prediction[0] == 1 else "closed"

while True:
    with frame_lock:
        img = current_frame.copy()
    if img is None:
        print("No frame available.")
        time.sleep(1)
        continue

    garage_door_state = get_garage_door_state(img)
    print("Garage door state:", garage_door_state)

    mqtt_client.publish(mqtt_queue, garage_door_state)
    time.sleep(5)