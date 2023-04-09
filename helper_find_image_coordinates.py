import cv2
import numpy as np

# Initialize the coordinates
x1, y1, x2, y2 = 200, 0, 500, 300
drawing = False

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        print(f"Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

def process_image(img):
    global x1, y1, x2, y2

    # Crop the image using the defined coordinates
    if x2 > x1 and y2 > y1:
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img
    else:
        return img

# Replace the URL with your RTSP URL
rtsp_url = 'rtsp://localhost:51610/a5793736882c5dbc'
cap = cv2.VideoCapture(rtsp_url)

# Create a window and set the mouse callback
cv2.namedWindow("res")
cv2.setMouseCallback("res", draw_rectangle)

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
