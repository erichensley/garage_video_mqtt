import cv2
import numpy as np

def process_image(img):
    # Define the top-left and bottom-right coordinates of the region of interest
    x1, y1, x2, y2 = 268, 1, 542, 112

    # Crop the image using the defined coordinates
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

# Replace the URL with your RTSP URL
rtsp_url = 'rtsp://localhost:51610/a5793736882c5dbc'
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to retrieve frame. Check your RTSP URL.")
        break

    img = process_image(img)

    cv2.imshow("res", img)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' key to exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




