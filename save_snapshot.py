import cv2
from datetime import datetime

def process_image(img):
    # Define the top-left and bottom-right coordinates of the region of interest
    x1, y1, x2, y2 = 268, 1, 542, 112

    # Crop the image using the defined coordinates
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

cap = cv2.VideoCapture('rtsp://localhost:51610/a5793736882c5dbc')
ret, img = cap.read()
img = process_image(img)
cv2.imwrite("snapshots/" + datetime.now().strftime("%m%d%Y%H%M%S") + ".jpg", img)
